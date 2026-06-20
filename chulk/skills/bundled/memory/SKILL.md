# Memory Skill

Use this skill when the user request involves saving, retrieving, summarizing, or updating durable information.

## Distinction

- A memory is a stored fact, preference, project note, or prior-work summary.
- This skill is a procedural playbook for deciding how to use memory.
- The memory store should not contain tool schemas, full skill instructions, or secrets.

Guidelines:

- Keep short-term conversation history separate from long-term memory.
- Search before assuming a memory exists.
- Save only durable, useful information.
- Avoid storing secrets or irrelevant details.
- Inject only relevant memories into the model prompt.
- Treat `persona`, `preference`, `style`, and `workflow` tags as profile context.
- Use profile memories to adapt tone, detail level, and task-solving style.
- Prefer updating or archiving duplicate/stale memories over creating noisy repeats.
- Use source and confidence metadata when the origin or certainty of a memory matters.
- Use `MEMORY.md` import/export for human review, not as the primary runtime memory store.
