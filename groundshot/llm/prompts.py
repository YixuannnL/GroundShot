"""All LLM/VLM prompt templates.

Design note (paper Sec. 4.6): to hit the reported call budget (~3.4 text LLM calls
per script), parsing + reference-quality estimation + layered-prompt construction
are BATCHED into one call per script (PARSE_*). Verification is risk-gated.
"""

# ------------------------------------------------------------------ parsing
PARSE_SYSTEM = """You are the script-analysis module of a multi-shot video generation system.
You must output ONLY a single valid JSON object, no prose."""

PARSE_USER = """Analyze this multi-shot video script.

GLOBAL CAPTION:
{global_caption}

SHOTS:
{shots_block}

{known_entities_block}

Produce a JSON object with EXACTLY these keys:

1. "style_layer": one sentence distilling ONLY the shared visual style, mood, lighting
   and color tone from the global caption. Exclude any character/object/location specifics.

2. "entities": object mapping stable entity IDs to records. IDs must be
   "char_*" for characters, "obj_*" for recurring objects, "loc_*" for locations.
   Link every textual mention (names, roles, pronouns like "he"/"the detective")
   of the same real-world entity to ONE id (cross-shot coreference).
   Each record: {{"type": "character|object|location",
                 "name": "<short grounding phrase, e.g. 'bald man in tan corduroy jacket'>",
                 "description": "<full appearance description for generation prompts>",
                 "aliases": ["..."],
                 "gender": "<characters only: male|female|null when stated or clearly implied>"}}
   Include ONLY recurring or visually important entities. Locations count as entities.

3. "shots": array, one record per shot, in order:
   {{"shot_id": <int>,
     "entity_ids": [ids of ALL entities visible in this shot, including its location],
     "expected_char_count": <int or null, explicit or strongly implied number of people>,
     "entity_layer": "<the FULL visual appearance of every non-location entity present
                      in THIS shot, restated from its entities record: physique, hair,
                      face, wardrobe for characters; appearance for objects. Pure
                      appearance, no actions or story. The video generator sees ONLY
                      the prompt, so omitting wardrobe here loses it in the output>",
     "action_layer": "<what happens: subjects, action, camera framing; self-contained,
                      no pronouns referring outside the shot>",
     "ref_quality": {{"<entity_id>": <float 0-1>}} }}

   "ref_quality" estimates, for each entity in the shot, how useful this shot is
   expected to be as a VISUAL REFERENCE SOURCE for that entity (Eq. 1 of GroundShot):
   - characters: near-frontal, unoccluded, close-up/medium framing, sharp, well-lit -> high;
     tiny in a wide shot, back view, occluded, motion-blurred, extreme lighting -> low.
   - objects: large, complete, unoccluded, canonical view -> high.
   - locations: wide/establishing views showing much of the environment with few
     foreground subjects -> high; close-ups with little visible background -> low.

Rules:
- Do NOT invent entities that the script does not mention.
- Every shot's entity_ids must be consistent with the coreference links.
- entity_layer must contain ONLY entities present in that shot (prevents leakage)."""

KNOWN_ENTITIES_BLOCK = """KNOWN ENTITY DEFINITIONS (authoritative; reuse these exact IDs,
you may refine names/descriptions but never merge or split them):
{entities_json}"""

# ---------------------------------------------------------- reference selection
SELECTOR_SYSTEM = """You select reference images for a reference-conditioned video generator.
Preserve identity/scene consistency while matching the target shot. Output ONLY JSON."""

SELECTOR_USER = """Entity: {entity_name} ({entity_type})
Description: {entity_desc}
Target shot (type: {shot_type}): {shot_text}

You are shown {n_images} candidate reference images in order.
Image 1 is the CANONICAL reference (identity anchor){aux_note}.

Choose which references best serve this target shot:
- characters: usually keep the canonical for identity; add one auxiliary only when it
  better matches the requested expression, pose, or viewing direction.
- objects/locations: may use auxiliary-only subsets when another view better matches
  the target framing, camera direction, or scene coverage.

Return JSON:
{{"use_canonical": <bool>,
  "selected_indices": [<0-based indices over the AUXILIARY candidates>],
  "selection_mode": "<canonical_only|canonical_plus_aux|aux_only|multi_aux|none>",
  "confidence": <float 0-1>,
  "reason": "<short>",
  "analysis": {{"candidate_1": "<short>", "...": "..."}},
  "alternatives": [<int>]}}"""

# ---------------------------------------------------------------- VLM critic
CRITIC_SYSTEM = """You are a strict quality critic for generated video shots.
You inspect sampled frames and report structured issues. Output ONLY JSON."""

CRITIC_USER = """A video shot was generated for this description:
"{shot_text}"

Expected entities in the shot:
{entities_block}
Expected number of people: {expected_count}

{ref_note}The following image tiles the shot's sampled frames in temporal order.

Evaluate and return JSON:
{{"quality": <float 0-1, overall fidelity to the description and visual quality>,
  "issues": [{{"type": "<identity_mismatch|count_error|missing_entity|extra_entity|
               clothing_mismatch|style_drift|lighting_inconsistency|unnatural_pose|
               motion_artifact|rendering_artifact|prompt_mismatch|quality_degradation>",
              "severity": "<minor|severe>",
              "entity_id": "<id or empty>",
              "detail": "<short>"}}],
  "notes": "<one sentence>"}}
Report an issue ONLY if clearly visible. count_error/missing_entity/identity_mismatch
against provided references are severe; mild stylistic wobble is minor."""

CRITIC_REF_NOTE = """Reference images for recurring entities are attached BEFORE the frame
tile, in this order: {ref_order}. Compare identities against them.
"""

# ------------------------------------------------------------ crop semantic check
CROP_CHECK_SYSTEM = """You validate whether a cropped image is a reliable visual reference.
Output ONLY JSON."""

CROP_CHECK_USER = """Entity: {entity_name} ({entity_type})
Description: {entity_desc}

Is this crop a reliable reference for the entity? Consider: does it actually show this
entity; is it recognizable and mostly unoccluded; for characters is the face usable;
any heavy blur, artifacts, or misleading content?

Return JSON: {{"match": <bool>, "reliability": <float 0-1>, "reason": "<short>"}}"""

SCENE_CHECK_USER = """Location entity: {entity_name}
Description: {entity_desc}

This image is a scene reference reconstructed by removing the foreground PEOPLE from a
generated frame. Validate it (Supp. 8.2): (a) unmasked background intact and matching
the description, (b) removed regions plausibly filled without obvious smears or blocky
artifacts, (c) no residual people or body parts remain.

Furniture, props, and objects that naturally belong to the location (desks, laptops,
plants, artwork...) are PART of the scene — never reject for them. Judge only removal
quality and background fidelity.

Return JSON: {{"valid": <bool>, "quality": <float 0-1>, "reason": "<short>"}}"""

# ---------------------------------------------------------------- repair prompt
REPAIR_HINTS = {
    "count_error": "IMPORTANT: the shot must contain EXACTLY {expected_count} people: "
                   "{char_names}. No additional people, none missing.",
    "missing_entity": "IMPORTANT: {entity_name} MUST be clearly visible in the shot.",
    "extra_entity": "IMPORTANT: show ONLY these subjects: {char_names}. No other people.",
    "style_drift": "Strictly keep this visual style: {style_layer}",
    "lighting_inconsistency": "Strictly keep this lighting and tone: {style_layer}",
    "clothing_mismatch": "Keep wardrobe EXACTLY as described: {entity_layer}",
    "prompt_mismatch": "Follow the action description precisely: {action_layer}",
}
