# Prompts for Stage 1: generateInteraction().

REASONING_PROMPT = """You are a clinical oncology pharmacist analyzing whether a therapeutic agent directly modulates a biological pathway in cancer.

**AGENT TO ANALYZE:** {agent_name}
**AGENT CATEGORY:** {agent_category}
**PATHWAY TO ANALYZE:** {pathway_name}

CRITICAL INSTRUCTIONS:
- You must ONLY analyze {agent_name}
- Do NOT mention, compare to, or discuss ANY other drugs
- Do NOT use examples from other agents
- If {agent_name} does NOT directly target {pathway_name}, simply say NO

Analyze whether {agent_name} has a **direct, clinically-validated interaction** with the {pathway_name} pathway.

Consider ONLY for {agent_name}:
1. What is {agent_name}'s primary molecular target?
2. Is that target a core component of the {pathway_name} pathway?
3. Does {agent_name} have clinical evidence (FDA approval OR Phase I+ trials)?
4. Is this the agent's PRIMARY mechanism of action?
5. In which cancer types is the target dysregulated (overexpressed/overactive/mutated/lost)?

Provide your analysis and conclusion for {agent_name} ONLY (YES or NO).

Begin analysis:"""


PARSING_PROMPT = """Extract structured data from this biomedical analysis.

Context: Determining if agent "{agent_name}" directly modulates pathway "{pathway_name}"

Analysis text:
{plaintext}

Extract and return JSON in this format:

{{
  "agentName": "{agent_name}",
  "pathwayName": "{pathway_name}",
  "hasInteraction": true/false,
  "agentEffect": "inhibits/activates/modulates" (if hasInteraction=true),
  "primaryTarget": "target name" (if hasInteraction=true),
  "cancerType": "cancer type" (if hasInteraction=true),
  "targetStatus": "overexpressed/overactive/mutated/lost" (if hasInteraction=true),
  "mechanismType": "direct/downstream" (if hasInteraction=true)
}}

Rules:
- Set hasInteraction=true ONLY if analysis concludes a valid, direct interaction
- Set hasInteraction=false if analysis says NO, insufficient evidence, indirect mechanism, or target not in pathway
- If hasInteraction=false, omit the optional fields
- If hasInteraction=true, all optional fields are required
- Return JSON only, no explanations

Output:"""