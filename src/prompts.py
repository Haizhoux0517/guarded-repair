REASONER_PROMPT = """
You are a careful mathematical reasoning assistant.

Solve the following grade-school math problem step by step.

You must output in this exact format:

Step 1: ...
Step 2: ...
Step 3: ...
Final Answer: <number>

Problem:
{problem}
"""


REPAIR_PROMPT = """
You are repairing an unreliable mathematical reasoning process.

You are given:
1. The original problem
2. The previous reasoning
3. A structured meta-diagnosis report

Use the diagnosis to correct the reasoning.

Important:
- Do not merely paraphrase the previous answer.
- Recompute the solution carefully.
- Make each calculation explicit.
- Avoid unsupported logical jumps.
- The final answer must follow from the reasoning steps.

You must output in this exact format:

Step 1: ...
Step 2: ...
Step 3: ...
Final Answer: <number>

Problem:
{problem}

Previous Reasoning:
{reasoning}

Meta-Diagnosis:
{diagnosis}
"""

SEMANTIC_VALIDATOR_PROMPT = """
You are a semantic reasoning validator for mathematical word problems.

Your task is NOT to solve the problem from scratch.
Your task is to check whether the given reasoning correctly interprets the problem statement.

You must detect semantic errors such as:
1. Misinterpreting comparative phrases, e.g., "three times more than", "twice as many", "how much more likely".
2. Ignoring whether a quantity applies per item, per person, per group, per day, or in total.
3. Incorrectly deciding whether the subject should be included in a count.
4. Incorrect interpretation of "remaining", "left", "equally", "combined", "each", "per", "additional", or "total".
5. Producing an answer format that does not match the problem requirement, e.g., integer vs decimal, dollars, percentage, number of days/weeks.
6. Reasoning that is arithmetic-consistent but semantically inconsistent with the problem.

Important:
- Do NOT use any gold answer.
- Do NOT assume the provided reasoning is correct just because the arithmetic is valid.
- Focus on whether the equations and operations match the meaning of the problem.
- If the problem wording is ambiguous, mark it as semantic_ambiguity.
- If the reasoning appears incomplete, mark it as incomplete_reasoning.
- If the reasoning uses all numbers but models the relationship incorrectly, mark it as semantic_relation_error.

Return ONLY a valid JSON object with this exact schema:

{{
  "is_semantically_valid": true or false,
  "semantic_score": a number between 0 and 1,
  "error_type": "none" or "semantic_relation_error" or "missing_constraint" or "semantic_ambiguity" or "answer_format_error" or "incomplete_reasoning",
  "needs_repair": true or false,
  "suspected_issue": "short description of the suspected issue",
  "explanation": "brief explanation"
}}

Problem:
{problem}

Reasoning:
{reasoning}
"""