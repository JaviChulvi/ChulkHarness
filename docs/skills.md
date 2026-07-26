# Skills

Skills are procedural instructions loaded into model context; they are not
callable tools and are not durable memory.

- Bundled skills ship under `chulk/skills/bundled/`.
- Project skills live under `<runtime_dir>/skills/`, normally `.chulk/skills/`.
- `Skills.only(...)` limits what may be selected.
- `Skills.pin(...)` always loads a chosen skill.

The registry reads lightweight metadata first, selects relevant skills for the
turn, and only then injects bounded `SKILL.md` content. Keep skill instructions
reviewable, avoid secrets, and treat third-party playbooks as untrusted prompt
content. Secret-free `.chulk/skills/` playbooks may be committed as declarative
project configuration; other `.chulk/` content remains ignored runtime state.
A skill cannot grant a capability or bypass tool permissions.

Portable hosted applications can publish immutable, versioned procedural
content with `PortableSkill`, `SkillPublicationManager`, and their sync or async
store protocols. Draft content cannot execute; rollback changes an audited
active pointer; revocation blocks new resolutions. See
[portable authoring](authoring.md#governed-portable-skills).

See `examples/13_per_agent_skills.py`, [configuration](configuration.md), and
[permissions](permissions.md).
