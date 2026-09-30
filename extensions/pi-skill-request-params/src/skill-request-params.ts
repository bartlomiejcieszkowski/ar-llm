/**
 * Skill Request Params Extension
 * By Andrea Richiardi
 *
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in favor of the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <https://unlicense.org>
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/**
 * Skill Request Parameters Extension
 *
 * Injects provider request parameters (temperature, top_p, top_k, etc.) on a
 * per-provider / per-model / per-skill basis.  All scoping lives under the
 * `providers` key — providers absent from the config are silently skipped so
 * their payloads are never touched.
 *
 * Resolution order (lowest → highest priority, merged via Object.assign):
 *   1. providers.<provider>.default                          — provider-wide base
 *   2. providers.<provider>.models.<modelId>.default        — model refinement
 *   3. providers.<provider>.skills.<skillName>              — skill override
 *   4. providers.<provider>.models.<modelId>.skills.<skill> — most specific
 *
 * When no skill is active only levels 1 & 2 apply.
 * If resolution produces an empty object the payload is left unchanged.
 *
 * Skill detection strategies (unchanged from v1):
 *   1. Explicit `/skill:name`  — detected in `input` handler (before expansion)
 *   2. Expanded `<skill name="...">` block — parsed from `event.prompt`
 *   3. LLM auto-invocation — last non-system message scanned for skill paths
 *
 * Config file: `<agent-dir>/ar-llm/skill-request-params.json`
 * Default locations (checked in order):
 *   - $PI_CODING_AGENT_DIR/ar-llm/skill-request-params.json
 *   - ~/.pi/agent/ar-llm/skill-request-params.json
 *
 * Skill names must match the `name` field in your SKILL.md frontmatter.
 * Test with:  pi -e skill-request-params.ts
 * Auto-load:  place in ~/.pi/agent/extensions/
 */

// ============================================================
// CONSTANTS
// ============================================================

/** Provider-specific keys for thinking token budgets (pass-through, no renaming). */
const THINKING_BUDGET_KEYS = ["thinking_budget_tokens", "thinking_token_budget"] as const;

// ============================================================
// TYPES
// ============================================================

interface ChatTemplateKwargs {
  enable_thinking?: boolean;
  preserve_thinking?: boolean;
}

/** Parameters injected into the provider request payload. */
interface SkillRequestParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  repetition_penalty?: number;
  frequency_penalty?: number;
  chat_template_kwargs?: ChatTemplateKwargs;
  // Provider-specific thinking budget keys (pass-through, no renaming)
  [key: string]: unknown;
}

/** A skill entry: required params plus optional name aliases. */
interface SkillRequestEntry {
  params: SkillRequestParams;
  aliases?: string[]; // additional skill names that share these params
}

/**
 * Config scoped to a specific model within a provider.
 * - `default`: applied to every request on this model (no skill active)
 * - `skills`:  per-skill overrides merged on top of `default`
 */
interface ProviderModelConfig {
  default?: SkillRequestParams;
  skills?: Record<string, SkillRequestEntry>;
}

/**
 * Config for a single provider.
 * Extends ProviderModelConfig so `default` and `skills` apply provider-wide,
 * and `models` allows per-model refinement on top.
 */
interface ProviderConfig extends ProviderModelConfig {
  models?: Record<string, ProviderModelConfig>;
}

/**
 * Root config shape.  All param scoping lives under `providers`.
 * Providers absent from this map are silently skipped — their payloads
 * are never touched.
 *
 * Example:
 * {
 *   "providers": {
 *     "lm-studio": {
 *       "default": { "temperature": 0.6 },
 *       "skills": {
 *         "clojure-coder": { "params": { "temperature": 0.1, "top_k": 40 } }
 *       },
 *       "models": {
 *         "qwen3-30b-a3b": {
 *           "default": { "top_k": 20 },
 *           "skills": {
 *             "clojure-coder": { "params": { "chat_template_kwargs": { "enable_thinking": false } } }
 *           }
 *         }
 *       }
 *     },
 *     "anthropic": {
 *       "default": { "temperature": 0.7 }
 *     }
 *   }
 * }
 */
interface SkillRequestConfig {
  providers: Record<string, ProviderConfig>;
}

/**
 * Resolve the directory containing the config file:
 *   1. PI_CODING_AGENT_DIR env var
 *   2. ~/.pi/agent
 */
function resolveConfigDir(): string {
  return process.env.PI_CODING_AGENT_DIR
    || path.join(os.homedir(), ".pi", "agent");
}

/**
 * Load and validate the skill-request-params.json config file.
 * Throws if the file doesn't exist or has an invalid shape.
 */
function loadConfig(): SkillRequestConfig {
  const dir = resolveConfigDir();
  const filePath = path.join(dir, "ar-llm", "skill-request-params.json");

  if (!fs.existsSync(filePath)) {
    throw new Error(
      `[skill-request-params] Config file not found at ${filePath}\n` +
      `Set PI_CODING_AGENT_DIR to point to the directory containing skill-request-params.json.`
    );
  }

  const raw = fs.readFileSync(filePath, "utf-8");
  const config: SkillRequestConfig = JSON.parse(raw);

  if (!config.providers || typeof config.providers !== "object") {
    throw new Error(`[skill-request-params] Config missing "providers" key in ${filePath}`);
  }

  return config;
}

let CONFIG: SkillRequestConfig | null;

try {
  CONFIG = loadConfig();
} catch (err: any) {
  console.error(err.message);
  console.error("[skill-request-params] Extension disabled due to config error.");
  CONFIG = null;
}

/**
 * Resolve the effective SkillRequestParams for a given provider / model / skill
 * by merging the four config levels from lowest to highest priority:
 *
 *   1. providers.<provider>.default
 *   2. providers.<provider>.models.<modelId>.default
 *   3. providers.<provider>.skills.<skillName>      (only when skillName != null)
 *   4. providers.<provider>.models.<modelId>.skills.<skillName>
 *
 * Returns undefined when the provider is not in config (payload untouched).
 * Returns an empty-ish object when the provider is configured but nothing
 * matched — callers should check Object.keys(result).length before applying.
 */
function resolveParams(
  provider: string | null,
  modelId: string | null,
  skillName: string | null
): SkillRequestParams | undefined {
  if (!provider) return undefined;

  const providerCfg = CONFIG!.providers[provider];
  if (!providerCfg) return undefined;

  // Expand aliases so skill lookup works for both primary names and aliases.
  // Build a flat alias→canonicalEntry map scoped to this provider (+ model).
  function findSkillEntry(
    skillsCfg: Record<string, SkillRequestEntry> | undefined,
    name: string
  ): SkillRequestParams | undefined {
    if (!skillsCfg || !name) return undefined;
    // Direct match
    if (skillsCfg[name]) return skillsCfg[name].params;
    // Alias match
    for (const entry of Object.values(skillsCfg)) {
      if (entry.aliases?.includes(name)) return entry.params;
    }
    return undefined;
  }

  const modelCfg = modelId ? providerCfg.models?.[modelId] : undefined;

  // Level 1: provider default
  const l1 = providerCfg.default ?? {};
  // Level 2: model default
  const l2 = modelCfg?.default ?? {};
  // Level 3: provider skill
  const l3 = skillName ? (findSkillEntry(providerCfg.skills, skillName) ?? {}) : {};
  // Level 4: model skill
  const l4 = skillName ? (findSkillEntry(modelCfg?.skills, skillName) ?? {}) : {};

  return Object.assign({}, l1, l2, l3, l4);
}

/**
 * Merges skill-specific request params into the provider payload.
 */
function applySkillParams(payload: any, params: SkillRequestParams): any {
  const result = { ...payload };

  if (params.temperature !== undefined) {
    result.temperature = params.temperature;
  }
  if (params.top_p !== undefined) {
    result.top_p = params.top_p;
  }
  if (params.top_k !== undefined) {
    result.top_k = params.top_k;
  }
  if (params.min_p !== undefined) {
    result.min_p = params.min_p;
  }
  if (params.presence_penalty !== undefined) {
    result.presence_penalty = params.presence_penalty;
  }
  if (params.repetition_penalty !== undefined) {
    result.repetition_penalty = params.repetition_penalty;
  }
  if (params.frequency_penalty !== undefined) {
    result.frequency_penalty = params.frequency_penalty;
  }
  if (params.chat_template_kwargs !== undefined) {
    result.chat_template_kwargs = params.chat_template_kwargs;
  }
  // Pass through provider-specific thinking budget keys as-is
  for (const key of THINKING_BUDGET_KEYS) {
    if (params[key] !== undefined) {
      result[key] = params[key];
    }
  }

  // Remove top-level enable_thinking only when explicitly controlling thinking
  if (params.chat_template_kwargs?.enable_thinking !== undefined) {
    delete result.enable_thinking;
  }

  return result;
}

function formatParams(params: SkillRequestParams): string {
  const parts: string[] = [];
  if (params.temperature !== undefined) parts.push(`t=${params.temperature}`);
  if (params.top_p !== undefined) parts.push(`pp=${params.top_p}`);
  if (params.top_k !== undefined) parts.push(`tk=${params.top_k}`);
  if (params.min_p !== undefined) parts.push(`mp=${params.min_p}`);
  if (params.presence_penalty !== undefined) parts.push(`pres=${params.presence_penalty}`);
  if (params.repetition_penalty !== undefined) parts.push(`rep=${params.repetition_penalty}`);
  if (params.frequency_penalty !== undefined) parts.push(`fp=${params.frequency_penalty}`);
  if (params.chat_template_kwargs) {
    const { enable_thinking, preserve_thinking } = params.chat_template_kwargs;
    if (enable_thinking === false) parts.push(`th=off`);
    else if (enable_thinking === true) {
      const budget = THINKING_BUDGET_KEYS.map(k => params[k]).find(v => v !== undefined);
      const suffix = budget !== undefined ? `(${budget}tok)` : "";
      parts.push(preserve_thinking ? `th=preserve${suffix}` : `th=on${suffix}`);
    }
  } else {
    const budget = THINKING_BUDGET_KEYS.map(k => params[k]).find(v => v !== undefined);
    if (budget !== undefined) parts.push(`budget=${budget}tok`);
  }
  return parts.join(" ");
}

/** Build a short context label for status bar and notifications. */
function formatLabel(
  provider: string | null,
  modelId: string | null,
  skillName: string | null,
  params: SkillRequestParams
): string {
  const scope = [provider ?? "?", modelId ?? "*"].join("/");
  const skill = skillName ? ` [${skillName}]` : "";
  return `⚡ ${scope}${skill}: ${formatParams(params)}`;
}

/**
 * Parse <skill name="..."> block from expanded prompt text.
 * Matches the format pi generates when expanding /skill:name commands:
 *   <skill name="clojure-coder" location="/path/to/SKILL.md">
 */
function parseSkillBlock(prompt: string): string | null {
  const match = prompt.match(/<skill name="([^"]+)"/);
  return match ? match[1] : null;
}

// ============================================================
// Extension entry point
// ============================================================

export default function (pi: ExtensionAPI) {
  if (!CONFIG) return;

  let activeSkill: string | null = null;
  let manualOverride = false;
  let activeProvider: string | null = null;
  let activeModelId: string | null = null;
  // Reverse lookup: skill file path -> skill name (populated in before_agent_start)
  let skillPathToName: Map<string, string> = new Map();

  const DEBUG = false;
  const AR_LLM_TMP = path.join(os.tmpdir(), "ar-llm");
  const DEBUG_LOG = path.join(AR_LLM_TMP, "skill-request-params.log");
  function log(msg: string) {
    if (!DEBUG) return;
    fs.mkdirSync(AR_LLM_TMP, { recursive: true });
    fs.appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`);
  }

  // Track active provider + model whenever the user switches model.
  pi.on("model_select", (event, _ctx) => {
    activeProvider = (event.model as any).provider ?? null;
    activeModelId = (event.model as any).id ?? null;
    log(`model_select: ${activeProvider}/${activeModelId}`);
  });

  // Detect /skill:name in raw input BEFORE expansion swallows it
  pi.on("input", (event, ctx) => {
    activeSkill = null;
    manualOverride = false;
    const match = event.text.match(/\/skill:([a-z0-9-]+)/);
    // Accept the skill name if the provider config knows about it, or if no
    // provider is active yet (we'll validate at request time).
    // Only set activeSkill if resolveParams would actually produce params for
    // this provider+skill combo — avoids noisy notifications for unconfigured skills.
    const wouldApply = match
      ? resolveParams(activeProvider, activeModelId, match[1])
      : undefined;
    if (match && wouldApply && Object.keys(wouldApply).length > 0) {
      activeSkill = match[1];
      manualOverride = true;
      log(`Manual override: ${activeSkill}`);
      ctx.ui.notify(`[skill-request-params] Manual override: ${activeSkill}`, "info");
    }
    return { action: "continue" };
  });

  // Build path->name lookup and detect explicit skill from expanded prompt.
  // Also initialise provider/model from ctx.model for session-restore cases
  // where model_select may not have fired.
  pi.on("before_agent_start", (event, ctx) => {
    if (!activeProvider && ctx.model) {
      activeProvider = (ctx.model as any).provider ?? null;
      activeModelId = (ctx.model as any).id ?? null;
      log(`before_agent_start: initialised provider=${activeProvider} model=${activeModelId}`);
    }

    const skills = event.systemPromptOptions?.skills;

    // Build reverse path->name lookup from loaded skills
    skillPathToName.clear();
    if (skills && Array.isArray(skills)) {
      for (const skill of skills) {
        const name = typeof skill === "string" ? skill : (skill.name ?? (skill as unknown as Record<string, unknown>)["id"] as string ?? "");
        const filePath = typeof skill === "object" ? (skill.filePath ?? (skill as unknown as Record<string, unknown>)["location"] as string ?? "") : "";
        if (name && filePath) {
          skillPathToName.set(filePath, name);
        }
      }
    }
    log(`before_agent_start: skillPathToName has ${skillPathToName.size} entries`);

    // Check for explicit skill invocation in expanded prompt
    // (pi expands /skill:name into <skill name="..."> before this event fires)
    const explicitSkill = parseSkillBlock(event.prompt);
    log(`before_agent_start: parseSkillBlock returned ${explicitSkill ?? "null"}`);
    log(`before_agent_start: prompt preview: ${event.prompt.substring(0, 300)}`);
    if (explicitSkill) {
      activeSkill = explicitSkill;
      manualOverride = true;
      log(`Explicit skill from prompt: ${activeSkill}`);
      return;
    }

    if (!manualOverride) {
      activeSkill = null;
    }
  });

  // Detect LLM auto-invocation: scan non-system messages for skill file references.
  // System prompt includes all skill paths in <available_skills>, so skip it.
  pi.on("before_provider_request", (event, ctx) => {
    if (!activeSkill) {
      const messages = (event.payload as any).messages ?? [];
      // Find the last non-system message (most recent assistant/tool response)
      let lastMsg: any = null;
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role !== "system") {
          lastMsg = messages[i];
          break;
        }
      }
      if (lastMsg) {
        const msgStr = JSON.stringify(lastMsg);
        for (const [skillPath, name] of skillPathToName) {
          if (msgStr.includes(skillPath)) {
            activeSkill = name;
            log(`LLM auto-invoked skill: ${activeSkill}`);
            break;
          }
        }
      }
    }

    // Resolve params through the 4-level hierarchy.
    const params = resolveParams(activeProvider, activeModelId, activeSkill);

    if (params && Object.keys(params).length > 0) {
      const modifiedPayload = applySkillParams(event.payload, params);
      const label = formatLabel(activeProvider, activeModelId, activeSkill, params);
      ctx.ui.setStatus("skill-params", label);
      ctx.ui.notify(`[skill-request-params] ${label}`, "info");
      log(`Applied: ${label}`);
      return modifiedPayload;
    }
    ctx.ui.setStatus("skill-params", undefined);
    return undefined;
  });

  pi.on("session_shutdown", () => {
    activeSkill = null;
    manualOverride = false;
    activeProvider = null;
    activeModelId = null;
    skillPathToName.clear();
  });
}
