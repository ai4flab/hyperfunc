from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from gepa import GEPAAdapter
from gepa.core.engine import GEPAEngine as GEPA

from .core import Example, HyperSystem


@dataclass
class GEPAPromptOptimizer:
    """
    Real GEPA-based prompt optimizer.

    Integrates with the `gepa` library to optimize prompts based on feedback
    from `HyperSystem.eval_on_examples`.
    """
    
    steps: int = 10
    population_size: int = 4
    model: str = "gpt-4o" # Model used for reflection/mutation

    def optimize(
        self,
        system: HyperSystem,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        
        # We need to optimize each HyperFunction's prompt independently (or jointly?)
        # For v0, let's just optimize them one by one if they have optimize_prompt=True
        
        for hf in system.hyperfunctions:
            if not hf.optimize_prompt:
                continue
                
            # Create an adapter for this specific function
            adapter = HyperFunctionGEPAAdapter(
                system=system,
                target_hf_name=hf.name,
                train_data=train_data,
                metric_fn=metric_fn
            )
            
            # Initialize GEPA
            gepa = GEPA(
                adapter=adapter,
                model=self.model,
                population_size=self.population_size,
            )
            
            # Run optimization
            best_prompt = gepa.optimize(
                initial_prompt=hf.get_prompt(),
                steps=self.steps
            )
            
            # Set the best prompt back to the function
            hf.set_prompt(best_prompt)


class HyperFunctionGEPAAdapter(GEPAAdapter):
    """
    Adapter to connect a specific HyperFunction to the GEPA optimization loop.
    """
    
    def __init__(
        self,
        system: HyperSystem,
        target_hf_name: str,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ):
        self.system = system
        self.target_hf_name = target_hf_name
        self.train_data = train_data
        self.metric_fn = metric_fn
        self.hf = system._by_name[target_hf_name]

    def evaluate(self, prompt: str) -> float:
        """
        Evaluate a candidate prompt.
        """
        # Temporarily set the prompt
        original_prompt = self.hf.get_prompt()
        self.hf.set_prompt(prompt)
        
        try:
            # Run evaluation on the whole system (other parts stay constant)
            score = self.system.eval_on_examples(self.train_data, self.metric_fn)
            return score
        finally:
            # Restore original prompt so we don't mess up state for other candidates
            # (GEPA might expect stateless evaluation or handle state itself, 
            # but safer to be side-effect free here until committed)
            self.hf.set_prompt(original_prompt)

    def get_trace(self, prompt: str) -> str:
        """
        Get execution trace/feedback for reflection.
        For now, we'll just run it on a few examples and return the inputs/outputs.
        """
        # Temporarily set prompt
        original_prompt = self.hf.get_prompt()
        self.hf.set_prompt(prompt)
        
        trace_lines = []
        
        # We'll use a subset of data for tracing to keep it short
        trace_examples = self.train_data[:3] 
        
        try:
            for ex in trace_examples:
                # We only care about the target function's input/output for the trace
                # But we have to run it via the system or directly?
                # If we run via system, we get the full flow.
                # But here we just want to see how THIS function behaved.
                
                # Let's call the function directly (it's bound to the system anyway)
                try:
                    output = self.hf(**ex.inputs)
                    trace_lines.append(f"Input: {ex.inputs}")
                    trace_lines.append(f"Output: {output}")
                    trace_lines.append(f"Expected: {ex.expected}")
                    trace_lines.append("---")
                except Exception as e:
                    trace_lines.append(f"Input: {ex.inputs}")
                    trace_lines.append(f"Error: {e}")
                    trace_lines.append("---")
                    
            return "\n".join(trace_lines)
            
        finally:
            self.hf.set_prompt(original_prompt)
