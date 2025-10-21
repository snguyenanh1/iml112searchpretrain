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

# Your task
- Suggest a plan to ensemble the {num_solutions} solutions. You should concentrate on how to merge, not the other parts like hyperparameters.
- The suggested plan should be novel, effective, and easy to implement.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur.
- Be specific about the ensemble method (weighted average, stacking, voting, etc.)."""

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

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur.
- Be specific about the ensemble method and how it differs from previous attempts."""

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
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {num_solutions} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the submission part due to formatting issue of submission.csv.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {num_solutions} provided solutions.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Save the final predictions to 'submission.csv' in the current directory."""

    def default_template(self) -> str:
        """Default ensemble prompt template"""
        return self._get_initial_plan_template()
