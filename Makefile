# Agent skill wiring
#
# `make codex`  links skills_studio/skills into .agents/skills (OpenAI Codex)
# `make claude` links skills_studio/skills into .claude/skills (Claude Code)
#               and skills_studio/agents into .claude/agents (Claude Code subagents)
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
	rm -rf .claude/agents
	ln -snf ../skills_studio/agents .claude/agents

clean-ai:
	rm -rf .agents/skills .claude/skills .claude/agents
