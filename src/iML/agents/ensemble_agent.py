"""Ensemble Agent for combining multiple iteration solutions.

This implementation is inspired by Google's ADK machine-learning-engineering ensemble agent.
It includes:
1. Initial plan generation
2. Plan implementation (code generation)
3. Code execution and scoring
4. Iterative refinement loop
"""

import logging
import json
import os
import shutil
import subprocess
import sys
import time
import re
import select
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_agent import BaseAgent
from ..prompts.ensemble_prompt import EnsemblePrompt
from .utils import init_llm

logger = logging.getLogger(__name__)


class EnsembleAgent(BaseAgent):
    """
    Agent to create ensemble strategies for combining multiple iteration solutions.
    Inspired by Google's ADK machine-learning-engineering ensemble approach.
    
    Workflow:
    1. Create workspace with data files
    2. Generate initial ensemble plan
    3. Implement plan into executable code
    4. Execute code and extract validation score
    5. Iteratively refine plan based on scores
    6. Select best ensemble and save submission_ensemble.csv
    """
    
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_refinement_rounds: int = 3):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="ensemble",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = EnsemblePrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_refinement_rounds = max_refinement_rounds
        self.ensemble_workspace = None
        self.ensemble_plans = []  # Store all plans tried
        self.ensemble_scores = []  # Store scores for each plan
        self.ensemble_codes = []  # Store codes for each plan

    def __call__(self, iteration_paths: List[str], iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create and execute ensemble strategy for multiple iterations.
        
        Args:
            iteration_paths: List of paths to iteration output folders
            iteration_results: List of iteration result dictionaries with metrics
            
        Returns:
            Dict containing ensemble results, best plan, and submission path
        """
        self.manager.log_agent_start("EnsembleAgent: Starting ensemble workflow...")
        
        if not iteration_paths or not iteration_results:
            logger.error("No iteration data provided for ensembling")
            return {"error": "No iteration data to ensemble"}
        
        # Filter only successful iterations
        successful_iterations = []
        successful_paths = []
        for path, result in zip(iteration_paths, iteration_results):
            if result.get("status") == "success":
                successful_iterations.append(result)
                successful_paths.append(path)
        
        if len(successful_iterations) < 2:
            logger.warning(f"Only {len(successful_iterations)} successful iteration(s). Ensemble requires at least 2.")
            return {
                "status": "skipped",
                "reason": "Not enough successful iterations for ensembling",
                "successful_count": len(successful_iterations)
            }
        
        logger.info(f"🎯 Creating ensemble from {len(successful_iterations)} successful iterations")
        
        # Step 1: Create workspace
        workspace_result = self._create_workspace(successful_paths)
        if "error" in workspace_result:
            return workspace_result
        self.ensemble_workspace = workspace_result["workspace_dir"]
        
        # Step 2: Extract codes from iterations
        iteration_codes = self._extract_codes(successful_paths, successful_iterations)
        
        # Step 3: Generate initial ensemble plan
        logger.info("📝 Step 1/4: Generating initial ensemble plan...")
        initial_plan = self._generate_initial_plan(successful_iterations, iteration_codes)
        if "error" in initial_plan:
            return initial_plan
        
        self.ensemble_plans.append(initial_plan["plan"])
        
        # Step 4: Implement and execute initial plan
        logger.info("⚙️  Step 2/4: Implementing and executing initial plan...")
        initial_result = self._implement_and_execute_plan(
            plan=initial_plan["plan"],
            iteration_codes=iteration_codes,
            iteration_results=successful_iterations,
            round_num=0
        )
        
        if "error" not in initial_result:
            self.ensemble_scores.append(initial_result.get("score", 0.0))
            self.ensemble_codes.append(initial_result.get("code", ""))
            logger.info(f"✅ Initial ensemble score: {initial_result.get('score', 'N/A')}")
        else:
            logger.warning(f"⚠️  Initial ensemble failed: {initial_result.get('error')}")
            self.ensemble_scores.append(float('inf'))  # Bad score
            self.ensemble_codes.append("")
        
        # Step 5: Iterative refinement loop
        logger.info(f"🔄 Step 3/4: Iterative refinement ({self.max_refinement_rounds} rounds)...")
        for round_num in range(1, self.max_refinement_rounds + 1):
            logger.info(f"   Round {round_num}/{self.max_refinement_rounds}")
            
            # Generate refined plan based on previous attempts
            refined_plan = self._generate_refined_plan(
                iteration_codes=iteration_codes,
                iteration_results=successful_iterations,
                round_num=round_num
            )
            
            if "error" in refined_plan:
                logger.warning(f"   Failed to generate refined plan: {refined_plan.get('error')}")
                continue
            
            self.ensemble_plans.append(refined_plan["plan"])
            
            # Implement and execute refined plan
            refined_result = self._implement_and_execute_plan(
                plan=refined_plan["plan"],
                iteration_codes=iteration_codes,
                iteration_results=successful_iterations,
                round_num=round_num
            )
            
            if "error" not in refined_result:
                score = refined_result.get("score", float('inf'))
                self.ensemble_scores.append(score)
                self.ensemble_codes.append(refined_result.get("code", ""))
                logger.info(f"   ✅ Refined ensemble score: {score}")
            else:
                logger.warning(f"   ⚠️  Refined ensemble failed: {refined_result.get('error')}")
                self.ensemble_scores.append(float('inf'))
                self.ensemble_codes.append("")
        
        # Step 6: Select best ensemble
        logger.info("🏆 Step 4/4: Selecting best ensemble...")
        best_result = self._select_best_ensemble()
        
        if best_result.get("best_idx") is not None:
            logger.info(f"✅ Best ensemble: Round {best_result['best_idx']} with score {best_result['best_score']}")
            
            # Copy best submission to main output folder
            submission_path = self._copy_best_submission(best_result)
            best_result["submission_path"] = submission_path
        else:
            logger.error("❌ No valid ensemble found")
        
        self.manager.log_agent_end("EnsembleAgent: Ensemble workflow completed.")
        return best_result
    
    def _create_workspace(self, iteration_paths: List[str]) -> Dict[str, Any]:
        """Create workspace directory and copy data files."""
        try:
            # Create ensemble directory structure
            ensemble_dir = Path(self.manager.output_folder) / "ensemble"
            if ensemble_dir.exists():
                shutil.rmtree(ensemble_dir)
            
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            input_dir = ensemble_dir / "input"
            input_dir.mkdir(exist_ok=True)
            final_dir = ensemble_dir / "final"
            final_dir.mkdir(exist_ok=True)
            
            # Copy data files from input folder
            input_data_path = Path(self.manager.input_data_folder)
            if input_data_path.exists():
                for file in input_data_path.iterdir():
                    if file.is_file() and not file.name.startswith('.'):
                        try:
                            shutil.copy2(file, input_dir / file.name)
                            logger.debug(f"Copied {file.name} to ensemble workspace")
                        except Exception as e:
                            logger.warning(f"Could not copy {file.name}: {e}")
            
            # Copy iteration submission files for reference
            for idx, iter_path in enumerate(iteration_paths, 1):
                submission_file = Path(iter_path) / "submission.csv"
                if submission_file.exists():
                    shutil.copy2(submission_file, input_dir / f"submission_iter{idx}.csv")
            
            logger.info(f"✅ Ensemble workspace created at {ensemble_dir}")
            
            return {
                "status": "success",
                "workspace_dir": str(ensemble_dir),
                "input_dir": str(input_dir),
                "final_dir": str(final_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to create ensemble workspace: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _extract_codes(self, iteration_paths: List[str], iteration_results: List[Dict]) -> Dict[str, str]:
        """Extract final executable code from each iteration."""
        codes = {}
        
        for path, result in zip(iteration_paths, iteration_results):
            iter_name = result.get("iteration_name", Path(path).name)
            
            # Try to find final executable code
            path_obj = Path(path)
            code_path = path_obj / "states" / "assemble" / "final_executable_code.py"
            if not code_path.exists():
                code_path = path_obj / "states" / "final_executable_code.py"
            
            if code_path.exists():
                try:
                    with open(code_path, 'r', encoding='utf-8') as f:
                        codes[iter_name] = f.read()
                    logger.debug(f"Extracted code from {iter_name}")
                except Exception as e:
                    logger.warning(f"Failed to read code from {code_path}: {e}")
                    codes[iter_name] = "# Code could not be read"
            else:
                logger.warning(f"Code file not found for {iter_name}")
                codes[iter_name] = "# Code file not found"
        
        return codes
    
    def _generate_initial_plan(self, iteration_results: List[Dict], iteration_codes: Dict[str, str]) -> Dict[str, Any]:
        """Generate initial ensemble plan using LLM."""
        try:
            prompt = self.prompt_handler.build(
                iteration_results=iteration_results,
                iteration_codes=iteration_codes,
                is_initial=True
            )
            
            self.manager.save_and_log_states(
                content=prompt,
                save_name="ensemble/round_0_plan_prompt.txt"
            )
            
            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name="ensemble/round_0_plan_response.txt"
            )
            
            # Extract plan from response
            plan = response.strip()
            
            return {
                "status": "success",
                "plan": plan
            }
            
        except Exception as e:
            logger.error(f"Failed to generate initial plan: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_refined_plan(self, iteration_codes: Dict[str, str], 
                              iteration_results: List[Dict], round_num: int) -> Dict[str, Any]:
        """Generate refined ensemble plan based on previous attempts."""
        try:
            prompt = self.prompt_handler.build_refinement_prompt(
                iteration_codes=iteration_codes,
                iteration_results=iteration_results,
                previous_plans=self.ensemble_plans,
                previous_scores=self.ensemble_scores,
                round_num=round_num
            )
            
            self.manager.save_and_log_states(
                content=prompt,
                save_name=f"ensemble/round_{round_num}_plan_prompt.txt"
            )
            
            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"ensemble/round_{round_num}_plan_response.txt"
            )
            
            plan = response.strip()
            
            return {
                "status": "success",
                "plan": plan
            }
            
        except Exception as e:
            logger.error(f"Failed to generate refined plan for round {round_num}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _implement_and_execute_plan(self, plan: str, iteration_codes: Dict[str, str],
                                   iteration_results: List[Dict], round_num: int) -> Dict[str, Any]:
        """Implement ensemble plan as code and execute it."""
        try:
            # Step 1: Generate implementation code
            impl_code = self._generate_implementation_code(plan, iteration_codes, iteration_results)
            
            if "error" in impl_code:
                return impl_code
            
            code = impl_code["code"]
            
            # Save the generated code
            self.manager.save_and_log_states(
                content=code,
                save_name=f"ensemble/round_{round_num}_ensemble_code.py"
            )
            
            # Step 2: Execute the code
            exec_result = self._execute_ensemble_code(code, round_num)
            
            if "error" in exec_result:
                return exec_result
            
            # Step 3: Extract score from output
            score = self._extract_score_from_output(exec_result["stdout"])
            
            return {
                "status": "success",
                "code": code,
                "score": score,
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"]
            }
            
        except Exception as e:
            logger.error(f"Failed to implement and execute plan: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_implementation_code(self, plan: str, iteration_codes: Dict[str, str],
                                     iteration_results: List[Dict]) -> Dict[str, Any]:
        """Generate executable Python code from ensemble plan."""
        try:
            prompt = self.prompt_handler.build_implementation_prompt(
                plan=plan,
                iteration_codes=iteration_codes,
                iteration_results=iteration_results
            )
            
            response = self.llm.assistant_chat(prompt)
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            if not code:
                return {
                    "status": "failed",
                    "error": "No valid Python code found in LLM response"
                }
            
            return {
                "status": "success",
                "code": code
            }
            
        except Exception as e:
            logger.error(f"Failed to generate implementation code: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code block with ```python
        if "```python" in response:
            start_idx = response.find("```python") + 9
            end_idx = response.find("```", start_idx)
            if end_idx != -1:
                return response[start_idx:end_idx].strip()
        
        # Try to find code block with just ```
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                # Take the first code block
                return parts[1].strip()
        
        return None
    
    def _execute_ensemble_code(self, code: str, round_num: int) -> Dict[str, Any]:
        """Execute ensemble code and capture output."""
        try:
            # Create execution directory
            exec_dir = Path(self.ensemble_workspace) / f"execution_round_{round_num}"
            exec_dir.mkdir(parents=True, exist_ok=True)
            
            # Write code to file
            code_file = exec_dir / "ensemble_code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing ensemble code for round {round_num}...")
            
            # Execute code
            working_dir = str(Path(self.ensemble_workspace))
            
            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=working_dir
            )
            
            stdout_chunks = []
            stderr_chunks = []
            
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
                    stderr_chunks.append(f"\nEnsemble execution reached time limit after {timeout} seconds.\n")
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
                    else:
                        stderr_chunks.append(line)
            
            # Ensure process exits
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            
            # Save outputs
            with open(exec_dir / "stdout.txt", 'w') as f:
                f.write(stdout)
            with open(exec_dir / "stderr.txt", 'w') as f:
                f.write(stderr)
            
            if process.returncode != 0:
                logger.error(f"Ensemble code execution failed with return code {process.returncode}")
                return {
                    "status": "failed",
                    "error": f"Execution failed: {stderr[:500]}"
                }
            
            # Check if submission.csv was created
            submission_file = Path(working_dir) / "submission.csv"
            if not submission_file.exists():
                submission_file = Path(working_dir) / "final" / "submission.csv"
            
            if submission_file.exists():
                # Copy to execution dir for archiving
                shutil.copy2(submission_file, exec_dir / "submission.csv")
            
            logger.info(f"✅ Ensemble code executed successfully")
            
            return {
                "status": "success",
                "stdout": stdout,
                "stderr": stderr,
                "submission_file": str(submission_file) if submission_file.exists() else None
            }
            
        except Exception as e:
            logger.error(f"Failed to execute ensemble code: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _extract_score_from_output(self, output: str) -> float:
        """Extract validation score from execution output."""
        # Look for patterns like:
        # "Final Validation Performance: 0.85"
        # "Validation Score: 0.85"
        # "Score: 0.85"
        # "RMSE: 0.15"
        # "Accuracy: 0.85"
        
        patterns = [
            r"Final Validation Performance:\s*([0-9.]+)",
            r"Validation Score:\s*([0-9.]+)",
            r"Ensemble Score:\s*([0-9.]+)",
            r"Final Score:\s*([0-9.]+)",
            r"Score:\s*([0-9.]+)",
            r"RMSE:\s*([0-9.]+)",
            r"Accuracy:\s*([0-9.]+)",
            r"F1[- ]Score:\s*([0-9.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    logger.info(f"Extracted score: {score}")
                    return score
                except ValueError:
                    continue
        
        logger.warning("Could not extract score from output, using default")
        return 0.0  # Default score if not found
    
    def _select_best_ensemble(self) -> Dict[str, Any]:
        """Select the best ensemble based on scores."""
        if not self.ensemble_scores:
            return {
                "status": "failed",
                "error": "No ensemble scores available"
            }
        
        # Filter out invalid scores (inf, nan)
        valid_indices = [
            i for i, score in enumerate(self.ensemble_scores)
            if score != float('inf') and score == score  # Check for not inf and not nan
        ]
        
        if not valid_indices:
            return {
                "status": "failed",
                "error": "No valid ensemble scores found"
            }
        
        # Find best score (assuming lower is better for errors, higher for accuracy)
        # This should be configurable based on metric type
        # For now, we'll assume lower is better (common for RMSE, MAE, etc.)
        best_idx = min(valid_indices, key=lambda i: self.ensemble_scores[i])
        best_score = self.ensemble_scores[best_idx]
        
        return {
            "status": "success",
            "best_idx": best_idx,
            "best_score": best_score,
            "best_plan": self.ensemble_plans[best_idx] if best_idx < len(self.ensemble_plans) else None,
            "best_code": self.ensemble_codes[best_idx] if best_idx < len(self.ensemble_codes) else None,
            "all_scores": self.ensemble_scores,
            "num_attempts": len(self.ensemble_scores)
        }
    
    def _copy_best_submission(self, best_result: Dict[str, Any]) -> Optional[str]:
        """Copy best ensemble submission to main output folder."""
        try:
            best_idx = best_result.get("best_idx")
            if best_idx is None:
                return None
            
            # Find submission file from best execution
            exec_dir = Path(self.ensemble_workspace) / f"execution_round_{best_idx}"
            submission_file = exec_dir / "submission.csv"
            
            if not submission_file.exists():
                # Try alternative location
                submission_file = Path(self.ensemble_workspace) / "submission.csv"
            
            if not submission_file.exists():
                logger.warning("Best ensemble submission.csv not found")
                return None
            
            # Copy to main output folder
            output_folder = Path(self.manager.output_folder)
            dest_file = output_folder / "submission_ensemble.csv"
            shutil.copy2(submission_file, dest_file)
            
            logger.info(f"✅ Best ensemble submission copied to {dest_file}")
            
            # Also save metadata
            metadata = {
                "best_round": best_idx,
                "best_score": best_result.get("best_score"),
                "best_plan": best_result.get("best_plan"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(output_folder / "ensemble_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return str(dest_file)
            
        except Exception as e:
            logger.error(f"Failed to copy best submission: {e}")
            return None
