"""Ensemble prompt for combining multiple iteration solutions.

Based on Google's ADK machine-learning-engineering ensemble prompts.
"""

import logging
from typing import Dict, List, Optional
from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class EnsemblePrompt(BasePrompt):
    """Prompt handler for ensemble agent"""

    def build(self, iteration_results: list, iteration_codes: dict, is_initial: bool = True) -> str:
        """
        Build ensemble prompt from iteration results.
        
        Args:
            iteration_results: List of iteration result dicts with metrics
            iteration_codes: Dict mapping iteration names to their final code
            is_initial: Whether this is the initial plan generation
        """
        num_solutions = len(iteration_results)
        
        # Format iteration solutions
        solutions_text = []
        for i, (result, (name, code)) in enumerate(zip(iteration_results, iteration_codes.items()), 1):
            status = result.get("status", "unknown")
            metrics = result.get("metrics", {})
            
            solution_block = f"""# Python Solution {i} ({name})
**Status**: {status}
**Metrics**: {metrics}

```python
{code}
```
"""
            solutions_text.append(solution_block)
        
        if is_initial:
            template = self._get_initial_plan_template()
        else:
            template = self.default_template()
        
        prompt = template.format(
            num_solutions=num_solutions,
            python_solutions="\n\n".join(solutions_text)
        )
        
        return prompt
    
    def build_refinement_prompt(self, iteration_codes: Dict[str, str], 
                                iteration_results: List[Dict],
                                previous_plans: List[str],
                                previous_scores: List[float],
                                round_num: int) -> str:
        """
        Build refinement prompt based on previous attempts.
        
        Args:
            iteration_codes: Dict of iteration codes
            iteration_results: List of iteration results
            previous_plans: List of previously tried plans
            previous_scores: List of scores for previous plans
            round_num: Current refinement round number
        """
        num_solutions = len(iteration_results)
        
        # Format iteration solutions
        solutions_text = []
        for i, (name, code) in enumerate(iteration_codes.items(), 1):
            result = iteration_results[i-1] if i-1 < len(iteration_results) else {}
            metrics = result.get("metrics", {})
            
            solution_block = f"""# Python Solution {i} ({name})
**Metrics**: {metrics}

```python
{code}
```
"""
            solutions_text.append(solution_block)
        
        # Format previous plans and scores
        prev_plans_text = []
        for idx, (plan, score) in enumerate(zip(previous_plans, previous_scores)):
            if score == float('inf') or score != score:  # inf or nan
                score_str = "Failed"
            else:
                score_str = f"{score:.5f}"
            
            prev_plans_text.append(f"""## Plan {idx}: 
{plan}

## Score: {score_str}
""")
        
        # Determine if lower or higher score is better
        # For now, assume lower is better (RMSE, MAE, etc.)
        valid_scores = [s for s in previous_scores if s != float('inf') and s == s]
        if valid_scores:
            criteria = "lower"  # Assuming lower is better
        else:
            criteria = "lower"
        
        template = self._get_refinement_template()
        
        prompt = template.format(
            num_solutions=num_solutions,
            python_solutions="\n\n".join(solutions_text),
            prev_plans_and_scores="\n\n".join(prev_plans_text),
            criteria=criteria
        )
        
        return prompt
    
    def build_implementation_prompt(self, plan: str, iteration_codes: Dict[str, str],
                                   iteration_results: List[Dict]) -> str:
        """
        Build implementation prompt to generate code from plan.
        
        Args:
            plan: The ensemble plan to implement
            iteration_codes: Dict of iteration codes
            iteration_results: List of iteration results
        """
        num_solutions = len(iteration_codes)
        
        # Format iteration solutions
        solutions_text = []
        for i, (name, code) in enumerate(iteration_codes.items(), 1):
            solution_block = f"""# Python Solution {i} ({name})
```python
{code}
```
"""
            solutions_text.append(solution_block)
        
        template = self._get_implementation_template()
        
        prompt = template.format(
            num_solutions=num_solutions,
            python_solutions="\n\n".join(solutions_text),
            plan=plan
        )
        
        return prompt

    def parse(self, response: Dict) -> Dict:
        """
        Parse LLM response for ensemble plan.
        
        Returns:
            Dict with 'plan' and optionally 'code' if implementation is included
        """
        try:
            content = response.get("content", "") if isinstance(response, dict) else response
            
            # Try to extract code block if present
            code = None
            if "```python" in content:
                start_idx = content.find("```python") + 9
                end_idx = content.find("```", start_idx)
                if end_idx != -1:
                    code = content[start_idx:end_idx].strip()
            
            # Extract plan (text before code or entire content)
            if code:
                plan_end = content.find("```python")
                plan = content[:plan_end].strip()
            else:
                plan = content.strip()
            
            return {
                "status": "success",
                "plan": plan,
                "code": code,
                "full_response": content
            }
            
        except Exception as e:
            logger.error(f"Failed to parse ensemble response: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "plan": "",
                "code": None
            }
    
    def _get_initial_plan_template(self) -> str:
        """Template for initial ensemble plan generation (inspired by ADK)."""
        return """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide {num_solutions} Python Solutions used for the competition.
- Your task is to propose a plan to ensemble the {num_solutions} solutions to achieve the best performance.

{python_solutions}

# Important Information
- The iteration submission files are available as `./input/submission_iter1.csv`, `./input/submission_iter2.csv`, etc.
- You can directly load and combine these prediction files instead of re-running the entire solutions.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.

# Your task
- Suggest a plan to ensemble the {num_solutions} solutions. You should concentrate on how to merge, not the other parts like hyperparameters.
- The suggested plan should be novel, effective, and easy to implement.
- Consider the performance of each solution when designing the ensemble strategy.
- The ensemble should compute a real validation score to measure performance.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur.
- Be specific about the ensemble method (weighted average, stacking, voting, etc.).
- Specify how to validate the ensemble performance on a validation set."""

    def _get_refinement_template(self) -> str:
        """Template for refining ensemble plan (inspired by ADK)."""
        return """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {num_solutions} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{python_solutions}

# Ensemble plans you have tried

{prev_plans_and_scores}

# Your task
- Suggest a better plan to ensemble the {num_solutions} solutions. You should concentrate on how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be different from the previous plans you have tried and should receive a {criteria} score.
- Learn from the failures and successes of previous attempts.
- Note: Scores marked as "Failed" indicate that either the code failed to execute or returned a placeholder/invalid score (0.0).

# Important Information
- The iteration submission files are available as `./input/submission_iter1.csv`, `./input/submission_iter2.csv`, etc.
- You can directly load and combine these prediction files for faster and more reliable ensembling.
- Ensure your plan includes proper validation score computation on real data, NOT placeholder values.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur.
- Be specific about the ensemble method and how it differs from previous attempts.
- Specify how to compute a real validation score to measure ensemble performance."""

    def _get_implementation_template(self) -> str:
        """Template for implementing ensemble plan (inspired by ADK)."""
        return """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {num_solutions} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{python_solutions}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- The iteration submission files are available as `./input/submission_iter1.csv`, `./input/submission_iter2.csv`, etc. You can load and ensemble these predictions directly.

# CRITICAL REQUIREMENTS for Validation Score:
- You MUST compute a REAL validation score on actual validation/test data, NOT a placeholder or dummy value.
- Split the training data into train/validation sets (e.g., 80/20 split) OR use the provided validation data if available.
- Compute the ensemble predictions on the validation set.
- Calculate the actual evaluation metric (RMSE, Accuracy, F1, etc.) on these predictions.
- The validation score must be a real numeric value computed from actual predictions, NOT 0.0, NOT a placeholder.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {num_solutions} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the submission part due to formatting issue of submission.csv.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {num_solutions} provided solutions.
- MANDATORY: Print the REAL validation performance metric using the EXACT format: 'Final Validation Performance: {{actual_score}}' where {{actual_score}} is computed from real validation data.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Save the final predictions to 'submission.csv' in the current directory (the ensemble workspace directory).
- Ensure the submission.csv file has the correct format matching the sample submission file."""

    def default_template(self) -> str:
        """Default ensemble prompt template"""
        return self._get_initial_plan_template()
