import logging
import sys
import os
import uuid
import subprocess
import json
import shutil
import time
import signal
import threading
import platform
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from ..agents import (
    DescriptionAnalyzerAgent,
    ProfilingAgent,
    ProfilingSummarizerAgent,
    ModelRetrieverAgent,
    GuidelineAgent,
    PreprocessingCoderAgent,
    ModelingCoderAgent,
    AssemblerAgent,
    ComparisonAgent,
    DebugAgent,
    EnsembleAgent,
)
from ..agents.comparison_agent import IterationResultExtractor
from ..llm import ChatLLMFactory

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

class IterationTimeoutError(Exception):
    """Custom exception for iteration timeout."""
    pass

def iteration_timeout_handler(signum, frame):
    """Signal handler for iteration timeout (Unix/Linux only)."""
    raise IterationTimeoutError("Iteration execution exceeded the time limit.")

class IterationTimer:
    """Cross-platform iteration timer using threading."""
    
    def __init__(self, timeout_seconds, callback):
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self.timer = None
        self.is_expired = False
    
    def start(self):
        """Start the timeout timer."""
        self.is_expired = False
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_occurred)
        self.timer.start()
    
    def cancel(self):
        """Cancel the timeout timer."""
        if self.timer:
            self.timer.cancel()
    
    def _timeout_occurred(self):
        """Internal method called when timeout occurs."""
        self.is_expired = True
        self.callback()
    
    def check_timeout(self):
        """Check if timeout has occurred."""
        if self.is_expired:
            raise IterationTimeoutError("Iteration execution exceeded the time limit.")


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder
        self.config = config

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.description_analyzer_agent = DescriptionAnalyzerAgent(
            config=config,
            manager=self,
            llm_config=self.config.description_analyzer,
        )
        self.profiling_agent = ProfilingAgent(
            config=config,
            manager=self,
        )
        self.profiling_summarizer_agent = ProfilingSummarizerAgent(
            config=config,
            manager=self,
            llm_config=self.config.profiling_summarizer,
        )
        self.model_retriever_agent = ModelRetrieverAgent(
            config=config,
            manager=self,
        )
        self.guideline_agent = GuidelineAgent(
            config=config,
            manager=self,
            llm_config=self.config.guideline_generator,
        )
        self.preprocessing_coder_agent = PreprocessingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.preprocessing_coder,
        )
        self.modeling_coder_agent = ModelingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.modeling_coder,
        )
        self.assembler_agent = AssemblerAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,
        )
        self.comparison_agent = ComparisonAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,  # Using same LLM config as assembler
        )
        self.ensemble_agent = EnsembleAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,  # Using same LLM config as assembler
            max_refinement_rounds=getattr(config, 'ensemble_max_rounds', 3)
        )
        # Initialize DebugAgent once for search-driven patching across phases
        self.debug_agent = DebugAgent(
            config=config,
            manager=self,
            max_rounds=10,
        )

        self.context = {
            "input_data_folder": input_data_folder,
            "output_folder": output_folder,
            
        }

    def get_iteration_timeout(self, iteration_type):
        """Get the execution timeout for a specific iteration type."""
        # Check if iteration_timeouts configuration exists
        if hasattr(self.config, 'iteration_timeouts') and self.config.iteration_timeouts:
            timeout = self.config.iteration_timeouts.get(iteration_type)
            if timeout:
                return timeout
        
        # Fallback to default_iteration_timeout if configured
        if hasattr(self.config, 'default_iteration_timeout'):
            return self.config.default_iteration_timeout
            
        # Final fallback to per_execution_timeout
        return self.config.per_execution_timeout
    
    def _run_iteration_with_timeout(self, iteration_type, iteration_timeout):
        """Run iteration with cross-platform timeout handling."""
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            # Use threading-based timeout for Windows
            timeout_occurred = threading.Event()
            
            def timeout_callback():
                timeout_occurred.set()
            
            timer = IterationTimer(iteration_timeout, timeout_callback)
            timer.start()
            
            try:
                # Run the iteration pipeline with periodic timeout checks
                success = self._run_iteration_pipeline_with_checks(iteration_type, timeout_occurred)
                return success
            finally:
                timer.cancel()
                
        else:
            # Use signal-based timeout for Unix/Linux
            original_handler = signal.signal(signal.SIGALRM, iteration_timeout_handler)
            signal.alarm(iteration_timeout)
            
            try:
                success = self._run_iteration_pipeline(iteration_type)
                return success
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
    
    def _run_iteration_pipeline_with_checks(self, iteration_type, timeout_occurred):
        """Run iteration pipeline with periodic timeout checks for Windows."""
        try:
            # Step 1a: For pretrained iteration, run model retrieval now; otherwise clear suggestions
            if iteration_type == "pretrained":
                if timeout_occurred.is_set():
                    raise IterationTimeoutError("Iteration timeout occurred before model retrieval")
                model_suggestions = self.model_retriever_agent()
                # Stop iteration immediately if retrieval failed or empty
                if (not model_suggestions) or ("error" in model_suggestions) or (not model_suggestions.get("sota_models")):
                    logger.error("Pretrained iteration aborted: SOTA search failed or returned no candidates.")
                    return False
                self.model_suggestions = model_suggestions
                logger.info("Model retrieval completed for pretrained iteration.")

                # Candidate-wise loop: for each SOTA model, run guideline -> preprocessing -> modeling -> assembly
                models = self.model_suggestions.get("sota_models", [])
                any_success = False
                parent_iter_dir = Path(self.output_folder)
                first_success_idx = None
                for idx, cand in enumerate(models, start=1):
                    if timeout_occurred.is_set():
                        raise IterationTimeoutError("Iteration timeout occurred before guideline generation")
                    # Narrow suggestions to a single candidate
                    self.model_suggestions = {"sota_models": [cand], "source": model_suggestions.get("source", "sota-search")}

                    # Prepare candidate-specific output directory and switch context
                    candidate_dir = parent_iter_dir / f"candidate_{idx}"
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    original_output_folder = self.output_folder
                    self.output_folder = str(candidate_dir)

                    # Guideline
                    guideline = self.guideline_agent(iteration_type=iteration_type)
                    if "error" in guideline:
                        logger.error(f"Guideline generation failed for candidate {idx}: {guideline['error']}")
                        # Restore output folder before continuing
                        self.output_folder = str(original_output_folder)
                        continue
                    self.guideline = guideline
                    try:
                        # Save guideline inside candidate states folder
                        self.save_and_log_states(json.dumps(guideline, ensure_ascii=False, indent=2), "guideline/guideline_response.json")
                    except Exception:
                        pass

                    # Preprocessing
                    if timeout_occurred.is_set():
                        raise IterationTimeoutError("Iteration timeout occurred before preprocessing")
                    preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
                    if preprocessing_code_result.get("status") == "failed":
                        logger.error(f"Preprocessing failed for candidate {idx}: {preprocessing_code_result.get('error')}")
                        # Restore output folder before continuing
                        self.output_folder = str(original_output_folder)
                        continue
                    self.preprocessing_code = preprocessing_code_result.get("code")

                    # Modeling
                    if timeout_occurred.is_set():
                        raise IterationTimeoutError("Iteration timeout occurred before modeling")
                    modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
                    if modeling_code_result.get("status") == "failed":
                        logger.error(f"Modeling failed for candidate {idx}: {modeling_code_result.get('error')}")
                        # Restore output folder before continuing
                        self.output_folder = str(original_output_folder)
                        continue
                    self.modeling_code = modeling_code_result.get("code")

                    # Assembly
                    if timeout_occurred.is_set():
                        raise IterationTimeoutError("Iteration timeout occurred before assembly")
                    assembler_result = self.assembler_agent(iteration_type=iteration_type)
                    if assembler_result.get("status") == "failed":
                        logger.error(f"Assembly failed for candidate {idx}: {assembler_result.get('error')}")
                        # Restore output folder before continuing
                        self.output_folder = str(original_output_folder)
                        continue
                    # Require submission.csv to consider candidate successful
                    cand_submission = candidate_dir / "submission.csv"
                    if cand_submission.exists():
                        any_success = True
                        # Copy submission back to iteration root for compatibility and archiving
                        try:
                            # archive as submission_cand_{idx}.csv at iteration root
                            dst_archive = parent_iter_dir / f"submission_cand_{idx}.csv"
                            shutil.copy2(cand_submission, dst_archive)
                            # set the first successful as iteration-level submission
                            if first_success_idx is None:
                                shutil.copy2(cand_submission, parent_iter_dir / "submission.csv")
                                first_success_idx = idx
                        except Exception as e:
                            logger.warning(f"Could not copy submission for candidate {idx}: {e}")
                    else:
                        logger.error(f"Candidate {idx} reported success but produced no submission.csv; treating as failure.")

                    # Restore output folder for next candidate
                    self.output_folder = str(original_output_folder)

                return any_success
            else:
                # Ensure other iterations are not influenced by retrieval results
                if hasattr(self, "model_suggestions"):
                    delattr(self, "model_suggestions")

            # Step 1b: Run guideline agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before guideline generation")
                
            guideline = self.guideline_agent(iteration_type=iteration_type)
            if "error" in guideline:
                logger.error(f"Guideline generation failed: {guideline['error']}")
                return False
            self.guideline = guideline
            logger.info(f"Guideline generated successfully for {iteration_type}.")

            # Step 2: Run Preprocessing Coder Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before preprocessing")
                
            preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
            if preprocessing_code_result.get("status") == "failed":
                logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
                return False
            self.preprocessing_code = preprocessing_code_result.get("code")
            logger.info("Preprocessing code generated and validated successfully.")

            # Step 3: Run Modeling Coder Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before modeling")
                
            modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
            if modeling_code_result.get("status") == "failed":
                logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
                return False
            self.modeling_code = modeling_code_result.get("code")
            logger.info("Modeling code generated successfully.")

            # Step 4: Run Assembler Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before assembly")
                
            assembler_result = self.assembler_agent(iteration_type=iteration_type)
            if assembler_result.get("status") == "failed":
                logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
                return False
            self.assembled_code = assembler_result.get("code")
            logger.info("Final script generated and executed successfully.")
            
            return True
            
        except IterationTimeoutError:
            raise  # Re-raise timeout error
        except Exception as e:
            logger.error(f"Error in iteration pipeline: {e}")
            return False

    def run_pipeline_partial(self, stop_after="guideline"):
        """Run pipeline up to a specific checkpoint."""
        logger.info(f"Starting partial AutoML pipeline (stop after: {stop_after})...")

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        if stop_after == "description":
            logger.info("Pipeline stopped after description analysis.")
            return True

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        if stop_after == "profiling":
            logger.info("Pipeline stopped after profiling.")
            return True

        # Step 3a: Summarize profiling via LLM
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

        # Step 3b: Retrieve pretrained model suggestions
        model_suggestions = self.model_retriever_agent()
        self.model_suggestions = model_suggestions

        if stop_after == "pre-guideline":
            # Save the default prompt template for editing
            self.save_default_guideline_prompt_template()
            logger.info("Pipeline stopped before guideline generation.")
            logger.info("You can now edit the guideline prompt template and resume from guideline generation.")
            return True

        # Step 3c: Run guideline agent
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return False
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        if stop_after == "guideline":
            logger.info("Pipeline stopped after guideline generation.")
            logger.info("You can now manually edit the guideline in the states folder.")
            return True

        logger.info("Partial AutoML pipeline completed successfully!")
        return True

    def load_checkpoint_state(self):
        """Load previously saved checkpoint state."""
        import json
        import os
        
        states_dir = os.path.join(self.output_folder, "states")
        
        # Debug: List all files in states directory
        if os.path.exists(states_dir):
            files_in_states = os.listdir(states_dir)
            logger.info(f"Files found in states directory: {files_in_states}")
        else:
            logger.warning(f"States directory does not exist: {states_dir}")
            return
        
        # Load description analysis
        desc_file = os.path.join(states_dir, "description_analyzer_response.json")
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.description_analysis = json.load(f)
                logger.info("Loaded description analysis from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load description analysis: {e}")
        else:
            logger.warning(f"Description analysis file not found: {desc_file}")
            # Initialize as None so we can check later
            self.description_analysis = None
        
        # Load profiling result
        prof_file = os.path.join(states_dir, "profiling_result.json")
        if os.path.exists(prof_file):
            with open(prof_file, 'r', encoding='utf-8') as f:
                self.profiling_result = json.load(f)
            logger.info("Loaded profiling result from checkpoint")
        
        # Load profiling summary  
        prof_sum_file = os.path.join(states_dir, "profiling_summary.json")
        if os.path.exists(prof_sum_file):
            with open(prof_sum_file, 'r', encoding='utf-8') as f:
                self.profiling_summary = json.load(f)
            logger.info("Loaded profiling summary from checkpoint")
        
        # Load model suggestions
        model_file = os.path.join(states_dir, "model_retrieval.json")
        if os.path.exists(model_file):
            with open(model_file, 'r', encoding='utf-8') as f:
                self.model_suggestions = json.load(f)
            logger.info("Loaded model suggestions from checkpoint")
        
        # Load guideline (might be manually edited)
        guideline_file = os.path.join(states_dir, "guideline", "guideline_response.json")
        if not os.path.exists(guideline_file):
            # Backward compatibility
            legacy = os.path.join(states_dir, "guideline_response.json")
            guideline_file = legacy if os.path.exists(legacy) else guideline_file
        if os.path.exists(guideline_file):
            with open(guideline_file, 'r', encoding='utf-8') as f:
                self.guideline = json.load(f)
            logger.info("Loaded guideline from checkpoint")

    def resume_pipeline_from_checkpoint(self, start_from="preprocessing"):
        """Resume pipeline from a specific checkpoint."""
        logger.info(f"Resuming AutoML pipeline from: {start_from}...")
        
        # Load previous state
        self.load_checkpoint_state()
        
        # Validate required states are loaded
        if not hasattr(self, 'description_analysis') or self.description_analysis is None:
            logger.error("Cannot resume: description_analysis not found or is None")
            logger.error("Make sure you have run the pipeline at least until the description analysis step")
            return False
        
        # For resume from guideline, we don't need existing guideline
        if start_from != "guideline" and (not hasattr(self, 'guideline') or self.guideline is None):
            logger.error(f"Cannot resume from {start_from}: guideline not found")
            logger.error("For this resume point, you need to have run until guideline generation")
            return False

        if start_from == "guideline":
            # Load custom prompt template if available
            self.update_guideline_prompt_template()
            
            # Check if we have necessary data for guideline generation
            if not hasattr(self, 'profiling_result') or not hasattr(self, 'model_suggestions'):
                logger.warning("Missing profiling or model suggestions data. Running those steps first...")
                
                # Re-run profiling if needed
                if not hasattr(self, 'profiling_result'):
                    profiling_result = self.profiling_agent()
                    if "error" in profiling_result:
                        logger.error(f"Data profiling failed: {profiling_result['error']}")
                        return False
                    self.profiling_result = profiling_result
                
                # Re-run profiling summary if needed
                if not hasattr(self, 'profiling_summary'):
                    profiling_summary = self.profiling_summarizer_agent()
                    if "error" in profiling_summary:
                        logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
                        return False
                    self.profiling_summary = profiling_summary
                
                # Re-run model retrieval if needed
                if not hasattr(self, 'model_suggestions'):
                    model_suggestions = self.model_retriever_agent()
                    self.model_suggestions = model_suggestions
            
            # Re-run guideline generation (useful after editing prompt)
            guideline = self.guideline_agent()
            if "error" in guideline:
                logger.error(f"Guideline generation failed: {guideline['error']}")
                return False
            self.guideline = guideline
            logger.info("Guideline regenerated successfully.")

        if start_from in ["guideline", "preprocessing"]:
            # Step 4: Run Preprocessing Coder Agent
            preprocessing_code_result = self.preprocessing_coder_agent()
            if preprocessing_code_result.get("status") == "failed":
                logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
                return False
            self.preprocessing_code = preprocessing_code_result.get("code")
            logger.info("Preprocessing code generated and validated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling"]:
            # Step 5: Run Modeling Coder Agent
            modeling_code_result = self.modeling_coder_agent()
            if modeling_code_result.get("status") == "failed":
                logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
                return False
            self.modeling_code = modeling_code_result.get("code")
            logger.info("Modeling code generated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling", "assemble"]:
            # Step 6: Run Assembler Agent
            assembler_result = self.assembler_agent()
            if assembler_result.get("status") == "failed":
                logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
                return False
            self.assembled_code = assembler_result.get("code")
            logger.info("Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")
        return True

    def update_guideline_prompt_template(self, new_template: str = None):
        """
        Update the guideline prompt template.
        If new_template is None, it will load from a file if it exists.
        """
        import os
        
        # Try to load from file first
        custom_prompt_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        if new_template is None and os.path.exists(custom_prompt_file):
            with open(custom_prompt_file, 'r', encoding='utf-8') as f:
                new_template = f.read()
            logger.info("Loaded custom guideline prompt from file")
        elif new_template is None:
            logger.info("No custom prompt template provided, using default")
            return
        
        # Update the template
        self.guideline_agent.prompt_handler.template = new_template
        logger.info("Guideline prompt template updated")
        
        # Save the template for reference
        self.save_and_log_states(new_template, "guideline_prompt_template_used.txt")

    def save_default_guideline_prompt_template(self):
        """Save the default guideline prompt template for editing."""
        default_template = self.guideline_agent.prompt_handler.default_template()
        template_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(default_template)
        
        logger.info(f"Default guideline prompt template saved to: {template_file}")
        logger.info("You can edit this file and resume from guideline generation.")
        return template_file

    def run_pipeline_multi_iteration(self):
        """Run the pipeline with 3 iterations for different algorithm approaches."""
        logger.info("Starting Multi-Iteration AutoML Pipeline...")
        
        # Store original output folder
        original_output_folder = self.output_folder
        
        # Define iterations
        iterations = [
            {
                "name": "pretrained",
                "folder": "iteration_1_pretrained", 
                "description": "Pretrained Models"
            },
            {
                "name": "traditional",
                "folder": "iteration_2_traditional",
                "description": "Traditional ML algorithms (XGBoost, LightGBM, CatBoost)"
            },
            {
                "name": "custom_nn", 
                "folder": "iteration_3_custom_nn",
                "description": "Custom Neural Networks"
            }
        ]
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            return
            
        # Run each iteration
        iteration_paths = []
        for i, iteration in enumerate(iterations, 1):
            iteration_timeout = self.get_iteration_timeout(iteration['name'])
            logger.info(f"=== Starting Iteration {i}: {iteration['description']} ===")
            logger.info(f"Timeout set for this iteration: {iteration_timeout} seconds ({iteration_timeout/60:.1f} minutes)")
            
            # Create iteration-specific output folder
            iteration_output = os.path.join(original_output_folder, iteration['folder'])
            os.makedirs(iteration_output, exist_ok=True)
            iteration_paths.append(iteration_output)
            
            # Temporarily change output folder for this iteration
            self.output_folder = iteration_output
            
            # Track iteration start time
            iteration_start_time = time.time()
            
            success = False
            try:
                # Run iteration with cross-platform timeout handling
                success = self._run_iteration_with_timeout(iteration['name'], iteration_timeout)
                
                # Calculate iteration duration
                iteration_duration = time.time() - iteration_start_time
                
                if success:
                    logger.info(f"=== Iteration {i} completed successfully in {iteration_duration:.1f} seconds ===")
                else:
                    logger.error(f"Iteration {i} ({iteration['name']}) failed after {iteration_duration:.1f} seconds!")
                    
            except IterationTimeoutError:
                iteration_duration = time.time() - iteration_start_time
                logger.warning(f"⏰ Iteration {i} ({iteration['name']}) timed out after {iteration_timeout} seconds!")
                logger.info(f"Moving to next iteration...")
                success = False
                
            except Exception as e:
                iteration_duration = time.time() - iteration_start_time
                logger.error(f"Iteration {i} ({iteration['name']}) failed with error: {e}")
                success = False
        
        # Restore original output folder
        self.output_folder = original_output_folder
        
        # Extract comprehensive results from all iterations
        logger.info("=== Extracting results from all iterations ===")
        extractor = IterationResultExtractor()
        iteration_results = []
        
        for iteration_path in iteration_paths:
            result = extractor.extract_from_iteration_folder(iteration_path)
            iteration_results.append(result)
            logger.info(f"Extracted results from {result['iteration_name']}: {result['status']}")
        
        # Use LLM to intelligently compare and rank iterations
        logger.info("=== LLM-based Intelligent Iteration Comparison ===")
        comparison_result = self.comparison_agent(
            iteration_results=iteration_results,
            original_task_description=self.description_analysis
        )
        
        if "error" in comparison_result:
            logger.error(f"LLM comparison failed: {comparison_result['error']}")
            logger.info("Falling back to basic selection...")
            best_iteration_name = self._fallback_selection(iteration_results)
        else:
            best_iteration_name = comparison_result.get('best_iteration', {}).get('name')
            
            # Save detailed comparison report
            comparison_file = os.path.join(original_output_folder, "llm_comparison_results.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM comparison report saved to: {comparison_file}")
        
        # Create and execute ensemble from successful iterations
        logger.info("=== Creating and Executing Ensemble ===")
        ensemble_result = self.ensemble_agent(
            iteration_paths=iteration_paths,
            iteration_results=iteration_results
        )
        
        if ensemble_result.get("status") != "skipped":
            if "error" not in ensemble_result:
                logger.info("✅ Ensemble workflow completed successfully")
                
                # Save detailed ensemble results
                ensemble_file = os.path.join(original_output_folder, "ensemble_results.json")
                # Remove code from result to avoid huge JSON (code is saved separately)
                ensemble_summary = {
                    k: v for k, v in ensemble_result.items() 
                    if k not in ['best_code']
                }
                with open(ensemble_file, 'w', encoding='utf-8') as f:
                    json.dump(ensemble_summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Ensemble results saved to: {ensemble_file}")
                
                # Save best ensemble code
                if ensemble_result.get("best_code"):
                    ensemble_code_file = os.path.join(original_output_folder, "best_ensemble_code.py")
                    with open(ensemble_code_file, 'w', encoding='utf-8') as f:
                        f.write(ensemble_result.get("best_code"))
                    logger.info(f"💡 Best ensemble code saved to: {ensemble_code_file}")
                
                # Check if submission_ensemble.csv was created
                ensemble_submission = os.path.join(original_output_folder, "submission_ensemble.csv")
                if os.path.exists(ensemble_submission):
                    logger.info(f"🎯 Ensemble submission created: {ensemble_submission}")
                    logger.info(f"📊 Ensemble score: {ensemble_result.get('best_score', 'N/A')}")
                else:
                    logger.warning("⚠️  submission_ensemble.csv was not created")
            else:
                logger.warning(f"Ensemble workflow encountered issues: {ensemble_result.get('error')}")
        else:
            logger.info(f"⏭️  Ensemble skipped: {ensemble_result.get('reason', 'Unknown reason')}")
        
        # Copy best submission to final_submission folder
        if best_iteration_name:
            # Resolve folder by normalizing name (case/format-insensitive)
            def _normalize(s: str) -> str:
                import re
                return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") if s else ""

            candidates = []
            try:
                from pathlib import Path as _P
                root = _P(original_output_folder)
                for d in root.iterdir():
                    if d.is_dir() and d.name.startswith("iteration_"):
                        candidates.append(d)
            except Exception:
                pass

            resolved_path = None
            norm_target = _normalize(best_iteration_name)
            for d in candidates:
                if d.name == best_iteration_name or d.name.lower() == best_iteration_name.lower() or _normalize(d.name) == norm_target:
                    resolved_path = str(d)
                    break

            best_iteration_path = resolved_path or os.path.join(original_output_folder, best_iteration_name)
            success = self._copy_best_submission(best_iteration_path, original_output_folder)
            
            if success:
                logger.info(f"✅ Best submission copied from {best_iteration_name}")
                if "error" not in comparison_result:
                    logger.info(f"📊 LLM Reasoning: {comparison_result.get('reasoning_summary', 'No reasoning provided')}")
            else:
                logger.error("❌ Failed to copy best submission")
        else:
            logger.error("❌ No best iteration selected")
        
        # Print summary
        successful_count = len([r for r in iteration_results if r.get('status') == 'success'])
        logger.info(f"📈 Summary: {successful_count}/{len(iterations)} iterations successful")
        if best_iteration_name:
            logger.info(f"🏆 LLM Selected Winner: {best_iteration_name}")
        
        logger.info("Multi-Iteration AutoML Pipeline completed!")
    
    def _fallback_selection(self, iteration_results: List[Dict]) -> str:
        """Fallback selection method when LLM comparison fails."""
        # Simple fallback: prefer successful iterations in priority order
        priority_order = ["iteration_1_pretrained", "iteration_2_traditional", "iteration_3_custom_nn"]

        successful_iterations = [
            r for r in iteration_results
            if r.get('status') == 'success'
        ]

        if not successful_iterations:
            logger.warning("No successful iterations found for fallback selection")
            return None

        # Select based on priority order
        for preferred_name in priority_order:
            for iteration in successful_iterations:
                if preferred_name in iteration.get('iteration_name', ''):
                    logger.info(f"Fallback selected: {iteration['iteration_name']}")
                    return iteration['iteration_name']

        # If no match, select first successful
        first_successful = successful_iterations[0]['iteration_name']
        logger.info(f"Fallback selected first successful: {first_successful}")
        return first_successful
    
    def _copy_best_submission(self, source_iteration_path: str, target_folder: str) -> bool:
        """Copy the best submission to final_submission folder."""
        try:
            source_path = Path(source_iteration_path)
            target_path = Path(target_folder) / "final_submission"
            
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Copy submission.csv
            source_submission = source_path / "submission.csv"
            target_submission = target_path / "submission.csv"
            
            if source_submission.exists():
                shutil.copy2(source_submission, target_submission)
                logger.info(f"Copied best submission from {source_submission} to {target_submission}")
                
                # Copy comparison metadata
                metadata = {
                    "source_iteration": source_path.name,
                    "copied_at": datetime.now().isoformat(),
                    "files_copied": ["submission.csv"]
                }
                metadata_file = target_path / "selection_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                logger.error(f"Source submission file not found: {source_submission}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying best submission: {e}")
            return False

    def run_pipeline_single_iteration(self, iteration_type: str):
        """Run the pipeline with a single specific iteration approach."""
        logger.info(f"Starting Single-Iteration AutoML Pipeline ({iteration_type})...")
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            return
        
        # Create iteration-specific output folder
        iteration_info = {
            "traditional": {"folder": "iteration_traditional", "description": "Traditional ML algorithms"},
            "custom_nn": {"folder": "iteration_custom_nn", "description": "Custom Neural Networks"}, 
            "pretrained": {"folder": "iteration_pretrained", "description": "Pretrained Models"}
        }
        
        info = iteration_info.get(iteration_type, {"folder": f"iteration_{iteration_type}", "description": iteration_type})
        iteration_output = os.path.join(self.output_folder, info['folder'])
        os.makedirs(iteration_output, exist_ok=True)
        
        # Store original and temporarily change output folder
        original_output_folder = self.output_folder
        self.output_folder = iteration_output
        
        # Get iteration timeout
        iteration_timeout = self.get_iteration_timeout(iteration_type)
        logger.info(f"=== Running {info['description']} ===")
        logger.info(f"Timeout set for this iteration: {iteration_timeout} seconds ({iteration_timeout/60:.1f} minutes)")
        
        # Track iteration start time
        iteration_start_time = time.time()
        
        success = False
        try:
            # Run iteration with cross-platform timeout handling
            success = self._run_iteration_with_timeout(iteration_type, iteration_timeout)
            
            iteration_duration = time.time() - iteration_start_time
            if success:
                logger.info(f"Single-iteration pipeline ({iteration_type}) completed successfully in {iteration_duration:.1f} seconds!")
            else:
                logger.error(f"Single-iteration pipeline ({iteration_type}) failed after {iteration_duration:.1f} seconds!")
                
        except IterationTimeoutError:
            iteration_duration = time.time() - iteration_start_time
            logger.warning(f"⏰ Single iteration ({iteration_type}) timed out after {iteration_timeout} seconds!")
            success = False
            
        except Exception as e:
            iteration_duration = time.time() - iteration_start_time
            logger.error(f"Single iteration ({iteration_type}) failed with error: {e}")
            success = False
            
        finally:
            # Restore original output folder
            self.output_folder = original_output_folder
    
    def _run_shared_analysis(self):
        """Run the shared analysis steps (description, profiling, summarization)."""
        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

    # Note: Model retrieval will be run only within the pretrained iteration.
        
        return True
    
    def _run_iteration_pipeline(self, iteration_type):
        """Run the pipeline for a specific iteration type."""
        # Step 1a: For pretrained iteration, per-candidate; otherwise normal flow
        if iteration_type == "pretrained":
            model_suggestions = self.model_retriever_agent()
            if (not model_suggestions) or ("error" in model_suggestions) or (not model_suggestions.get("sota_models")):
                logger.error("Pretrained iteration aborted: SOTA search failed or returned no candidates.")
                return False
            self.model_suggestions = model_suggestions
            logger.info("Model retrieval completed for pretrained iteration.")

            models = self.model_suggestions.get("sota_models", [])
            any_success = False
            parent_iter_dir = Path(self.output_folder)
            first_success_idx = None
            for idx, cand in enumerate(models, start=1):
                # Narrow suggestions to a single candidate
                self.model_suggestions = {"sota_models": [cand], "source": model_suggestions.get("source", "sota-search")}
                # Prepare candidate-specific output directory and switch context
                candidate_dir = parent_iter_dir / f"candidate_{idx}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                original_output_folder = self.output_folder
                self.output_folder = str(candidate_dir)
                # Guideline
                guideline = self.guideline_agent(iteration_type=iteration_type)
                if "error" in guideline:
                    logger.error(f"Guideline generation failed for candidate {idx}: {guideline['error']}")
                    # Restore output folder before continuing
                    self.output_folder = str(original_output_folder)
                    continue
                self.guideline = guideline
                try:
                    # Save guideline inside candidate states folder
                    self.save_and_log_states(json.dumps(guideline, ensure_ascii=False, indent=2), "guideline/guideline_response.json")
                except Exception:
                    pass

                # Preprocessing
                preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
                if preprocessing_code_result.get("status") == "failed":
                    logger.error(f"Preprocessing failed for candidate {idx}: {preprocessing_code_result.get('error')}")
                    # Restore output folder before continuing
                    self.output_folder = str(original_output_folder)
                    continue
                self.preprocessing_code = preprocessing_code_result.get("code")

                # Modeling
                modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
                if modeling_code_result.get("status") == "failed":
                    logger.error(f"Modeling failed for candidate {idx}: {modeling_code_result.get('error')}")
                    # Restore output folder before continuing
                    self.output_folder = str(original_output_folder)
                    continue
                self.modeling_code = modeling_code_result.get("code")

                # Assembly
                assembler_result = self.assembler_agent(iteration_type=iteration_type)
                if assembler_result.get("status") == "failed":
                    logger.error(f"Assembly failed for candidate {idx}: {assembler_result.get('error')}")
                    # Restore output folder before continuing
                    self.output_folder = str(original_output_folder)
                    continue
                any_success = True
                # Copy submission back to iteration root for compatibility and archiving
                try:
                    cand_submission = candidate_dir / "submission.csv"
                    if cand_submission.exists():
                        # archive as submission_cand_{idx}.csv at iteration root
                        dst_archive = parent_iter_dir / f"submission_cand_{idx}.csv"
                        shutil.copy2(cand_submission, dst_archive)
                        # set the first successful as iteration-level submission
                        if first_success_idx is None:
                            shutil.copy2(cand_submission, parent_iter_dir / "submission.csv")
                            first_success_idx = idx
                except Exception as e:
                    logger.warning(f"Could not copy submission for candidate {idx}: {e}")
                # Restore output folder for next candidate
                self.output_folder = str(original_output_folder)

            return any_success

        else:
            if hasattr(self, "model_suggestions"):
                delattr(self, "model_suggestions")

        # Step 1b: Run guideline agent with iteration-specific algorithm constraint
        guideline = self.guideline_agent(iteration_type=iteration_type)
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return False
        self.guideline = guideline
        logger.info(f"Guideline generated successfully for {iteration_type}.")

        # Step 2: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return False
        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 3: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return False
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully.")

        # Step 4: Run Assembler Agent
        assembler_result = self.assembler_agent(iteration_type=iteration_type)
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return False
        self.assembled_code = assembler_result.get("code")
        logger.info("Final script generated and executed successfully.")
        
        return True

    def run_pipeline(self):
        """Run the entire pipeline from description analysis to code generation."""

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return
        logger.info(f"Analysis result: {analysis_result}")

        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return
        
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3: Run guideline agent
        # 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return
        self.profiling_summary = profiling_summary

        # 3b: Retrieve pretrained model/embedding suggestions
        model_suggestions = self.model_retriever_agent()
        self.model_suggestions = model_suggestions

        # 3c: Run guideline agent with summarized profiling + model suggestions
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return
        
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        # Step 4: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent()
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return

        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 5: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent()
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return
            
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully (not yet validated).")

        # Step 6: Run Assembler Agent to assemble, finalize, and run the code
        assembler_result = self.assembler_agent()
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return
        
        self.assembled_code = assembler_result.get("code")
        logger.info(f"Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def execute_code(self, code_to_execute: str, phase_name: str, attempt: int) -> dict:
        """
        Executes a string of Python code in a subprocess and saves the script,
        stdout, and stderr to a structured attempts folder.

        Args:
            code_to_execute: The Python code to run.
            phase_name: The name of the phase (e.g., "preprocessing", "assemble").
            attempt: The retry attempt number.

        Returns:
            A dictionary with execution status, stdout, and stderr.
        """
        # Create a structured directory for this attempt (under states)
        attempt_dir = Path(self.output_folder) / "states" / phase_name / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths for the script, stdout, and stderr
        script_path = attempt_dir / "generated_code.py"
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"

        # Write the code to the script file
        self.write_code_script(code_to_execute, str(script_path))

        logger.info(f"Executing code from: {script_path}")

        try:
            # Execute the script using subprocess with live streaming
            import select
            # Run from the dataset root so relative paths from description analyzer resolve
            working_dir = str(Path(self.input_data_folder).parent)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=working_dir,
            )

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            streams = [process.stdout, process.stderr]
            start_time = time.time()
            timeout = self.config.per_execution_timeout

            while streams:
                elapsed = time.time() - start_time
                remaining = max(0, timeout - elapsed)
                if remaining <= 0:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    stderr_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                    logger.error(f"Process reached time limit after {timeout} seconds.")
                    break

                readable, _, _ = select.select([s for s in streams if s], [], [], min(1, remaining))
                if not readable and process.poll() is not None:
                    break
                for stream in readable:
                    line = stream.readline()
                    if not line:
                        streams.remove(stream)
                        continue
                    if stream is process.stdout:
                        stdout_chunks.append(line)
                        logger.detail(line.rstrip())
                    else:
                        stderr_chunks.append(line)
                        logger.detail(line.rstrip())

            # Ensure process exits
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stderr_chunks.append("Process forcibly terminated after timeout\n")

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

            # Save outputs
            with open(stdout_path, "w") as f:
                f.write(stdout)
            with open(stderr_path, "w") as f:
                f.write(stderr)

            if process.returncode == 0:
                logger.info("Code executed successfully.")
                return {"success": True, "stdout": stdout, "stderr": stderr}
            else:
                logger.error(f"Code execution failed with return code {process.returncode}.")
                full_error = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                return {"success": False, "stdout": stdout, "stderr": full_error}
        except Exception as e:
            logger.error(f"An exception occurred during code execution: {e}")
            with open(stderr_path, "w") as f:
                f.write(str(e))
            return {"success": False, "stdout": "", "stderr": str(e)}


    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code_old(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", add_uuid=False)
        self.save_and_log_states(stdout, "stdout", add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, add_uuid=False):
        """
        Save content under output_folder/states. save_name can include nested folders
        like "guideline/guideline_prompt.txt".

        - When add_uuid is True, append a short UUID before the file extension.
        - Content may be a list or string; None is saved as "<None>".
        """
        # Optionally add a short UUID suffix to the filename
        if add_uuid:
            name, ext = os.path.splitext(save_name)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Compose full path and ensure parent directories exist
        states_dir = os.path.join(self.output_folder, "states")
        output_file = os.path.join(states_dir, save_name)
        parent_dir = os.path.dirname(output_file)
        os.makedirs(parent_dir, exist_ok=True)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is None:
                file.write("<None>")
            elif isinstance(content, list):
                file.write("\n".join(str(item) for item in content))
            else:
                file.write(content)

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
