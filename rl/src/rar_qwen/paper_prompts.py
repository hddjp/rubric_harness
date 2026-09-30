"""Runtime prompt reproduced verbatim from Appendix A.6 of the paper.

Source: Rubrics as Rewards, arXiv:2507.17746v2.
Do not paraphrase these strings: experiments rely on the published wording.
"""

RAR_IMPLICIT_SYSTEM = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.

Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.

Start your response with a valid JSON object that starts with "```json" and ends with "```".
The JSON object should contain a single key "rating" and the value should be an integer between 1 and 10.

Example response:
```json
{
  "rating": 7
}```"""

RAR_IMPLICIT_USER = """Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.


<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

Your JSON Evaluation:"""
# The released datasets already contain the paper-generated rubrics. Reproduction
# consumes those directly, so no rubric-generation prompt is executed locally.
