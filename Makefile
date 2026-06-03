# Agent skill wiring
#
# `make codex`  links skills_studio/skills into .agents/skills (OpenAI Codex)
# `make claude` links skills_studio/skills into .claude/skills (Claude Code)
# `make clean-ai` removes the generated links

.PHONY: codex claude clean-ai

codex:
	mkdir -p .agents
	rm -rf .agents/skills
	ln -snf ../skills_studio/skills .agents/skills

claude:
	mkdir -p .claude
	rm -rf .claude/skills
	ln -snf ../skills_studio/skills .claude/skills

clean-ai:
	rm -rf .agents/skills .claude/skills
