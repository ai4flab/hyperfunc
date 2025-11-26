import random
from pydantic import BaseModel
from hyperfunc import Example, HyperSystem, LMParam, ESHybridSystemOptimizer, GEPAPromptOptimizer, hyperfunction

# Mock LLM call for demo purposes
def mock_llm_call(prompt: str, temperature: float) -> str:
    """
    Simulates an LLM that extracts invoices.
    It's "good" if the prompt mentions "JSON" and temperature is low.
    """
    if "JSON" not in prompt:
        return "I don't know how to format this."
    
    if temperature > 0.5:
        return '{"invoice_number": "123", "total": "lots"}' # Bad format due to high temp
        
    return '{"invoice_number": "123", "date": "2024-12-01", "total": 99.95}'


class Invoice(BaseModel):
    invoice_number: str | None = None
    date: str | None = None
    total: float | None = None


@hyperfunction(
    model="mock-gpt",
    hp_type=LMParam,
    optimize_prompt=True,
    optimize_hparams=True,
)
def extract_invoice(doc: str, hp: LMParam) -> Invoice:
    """
    Extract invoice details.
    """
    # In a real app, you'd call an LLM here.
    # We simulate it:
    raw_json = mock_llm_call(extract_invoice.__doc__, hp.temperature)
    
    try:
        return Invoice.model_validate_json(raw_json)
    except Exception:
        return Invoice() # Return empty on failure


def run_demo():
    print("--- Starting HyperFunctions Demo ---")
    
    # 1. Data
    train_data = [
        Example(
            fn_name="extract_invoice",
            inputs={"doc": "INVOICE #123..."},
            expected=Invoice(invoice_number="123", date="2024-12-01", total=99.95),
        )
    ]

    def metric_fn(preds, expected):
        # Simple exact match metric
        score = 0.0
        for p, e in zip(preds, expected):
            if p == e:
                score += 1.0
        return score / len(preds)

    # 2. System
    # Note: We use a mock GEPA optimizer here because we don't have a real OpenAI key set up
    # for the demo to run automatically. In real usage, you'd use GEPAPromptOptimizer.
    
    class MockGEPAPromptOptimizer:
        def optimize(self, system, train_data, metric_fn):
            print("Running Mock GEPA...")
            # Simulate finding a better prompt
            hf = system._by_name["extract_invoice"]
            hf.set_prompt(hf.get_prompt() + " Please output valid JSON.")
            
    system = HyperSystem(
        prompt_optimizer=MockGEPAPromptOptimizer(),
        system_optimizer=ESHybridSystemOptimizer(steps=10, lr=0.1),
    )
    system.register_hyperfunction(extract_invoice)

    # 3. Baseline
    print("Baseline score:", system.evaluate(train_data, metric_fn))
    
    # 4. Optimize
    print("Optimizing...")
    system.optimize(train_data, metric_fn)
    
    # 5. Result
    print("Final score:", system.evaluate(train_data, metric_fn))
    print("Final Prompt:", extract_invoice.get_prompt())
    state = system.get_hp_state()
    hp_vec = state["extract_invoice"]
    print("Final Params (Temp):", hp_vec[0].item())


if __name__ == "__main__":
    run_demo()
