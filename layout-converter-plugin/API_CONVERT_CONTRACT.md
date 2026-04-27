# `POST /api/convert` — request body contract

The Figma plugin sends **`Content-Type: application/json`** with a **single JSON object**.  
All large binary fields are **one layer of standard Base64** (RFC 4648) over **raw bytes**.  
**No** `data:image/png;base64,` prefix. **No** JSON string nested inside another JSON string for the same payload.

Encoding rule: **`btoa`-style / standard Base64 of a `Uint8Array` as binary octets** — not UTF-8 encoding of a text string.

**Raster payload:** the plugin sends **exactly two** Base64-encoded PNG file blobs as strings:

1. **`banner_png_base64`** — full frame.  
2. **`element_atlas_png_base64`** — one packed image of all exported **leaf** layers (see below).

There is **no** `image_library_png_base64` map and **no** per-leaf `element_pngs[]` array.

---

## Fields

### `banner_png_base64` (string, required)

- **Content:** PNG file bytes of the **entire selected frame** at export scale 1.
- **Source:** `node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 1 } })` on the root frame, then Base64-encode the returned `Uint8Array`.
- **Guarantee:** Valid PNG (includes PNG signature `89 50 4E 47 0D 0A 1A 0A` after decode).

---

### `raw_json` (object, required)

- **Content:** Serialized Figma subtree for the selected frame (same tree the plugin uses for semantics): `id`, `path`, `name`, `type`, `bounds`, `visible`, `opacity`, optional `characters`, `layoutMode`, `children`, etc.
- **Not** double-encoded: this is a normal JSON object, not a string containing JSON.
- **Atlas alignment:** On nodes that were packed into the element atlas, the plugin adds **`atlas_region`**: `{ x, y, width, height }` in **atlas pixel space** (same `path` / `name` as the rest of the tree). Use it with `element_atlas_png_base64` to crop each leaf’s raster.

---

### `target_width` / `target_height` (number, required)

Logical target dimensions for conversion (may match frame size or a preset).

---

### `mode` (string, required)

Conversion mode, e.g. `apply_to_clone_fast`, `apply_to_clone_vlm`, `full_layout_debug`.

---

### `use_qwen` (boolean, required)

Whether the VLM path is requested (aligned with `mode`).

---

### `qwen_mode` (string, optional)

Present only when the UI selects VLM mode; e.g. `scene_only`.

---

### `element_image_refs` (array, optional)

Each row marks **where** an `IMAGE` fill/stroke appears in the tree. The plugin **does not** send a separate bitmap per `image_hash`; appearance is in **`banner_png_base64`** and, for **leaf** nodes with an image, typically also in a crop of **`element_atlas_png_base64`** at the matching **`path`** (`atlas_region` on `raw_json`).

| Property      | Type   | Meaning |
|---------------|--------|---------|
| `path`        | string | Index path aligned with `raw_json` (`"0"`, `"0/2"`, …). |
| `node_id`     | string | Figma node id at export time. |
| `name`        | string | Node name. |
| `type`        | string | Node type (e.g. `RECTANGLE`). |
| `image_hash`  | string | Figma internal image hash (identifier only). |
| `fill_role`   | string | `"fill"` or `"stroke"`. |

---

### `element_atlas_png_base64` (string, optional)

- **Content:** A **single** PNG whose canvas packs **rasterized leaves** (visible leaves, `INSTANCE` as one leaf, capped count). Row-packed layout with a **fixed pixel gap** between cells (see `ATLAS_GAP` in `code.js`); coordinates in `element_atlas_regions` and on `raw_json` as `atlas_region`.
- **Encoding:** Same Base64 rules as `banner_png_base64`.
- **Empty atlas:** If there are no qualifying leaves, this is **`""`** and `element_atlas_regions` is `[]` — skip PNG decode for the atlas.

---

### `element_atlas_regions` (array, optional)

One row per packed leaf, aligned with `raw_json` by **`path`**:

| Property        | Type   | Meaning |
|-----------------|--------|---------|
| `path`          | string | Same path convention as `raw_json`. |
| `node_id`       | string | Figma node id at export time. |
| `name`          | string | Node name. |
| `type`          | string | Node type. |
| `atlas_x`       | number | Left edge in atlas PNG pixels. |
| `atlas_y`       | number | Top edge in atlas PNG pixels. |
| `atlas_width`   | number | Width in atlas PNG pixels. |
| `atlas_height`  | number | Height in atlas PNG pixels. |

---

### Count helpers (integer, optional)

- `element_atlas_regions_count` — `element_atlas_regions.length`
- `element_image_refs_count` — `element_image_refs.length`

---

## Backend checklist

1. Accept all keys above on your Pydantic model with **`extra` not set to `"forbid"`** (this repo uses **`extra="allow"`** on `ConvertRequest`) so optional / future keys are not rejected.
2. Decode Base64 → **binary** → verify PNG for **`banner_png_base64`** and for **`element_atlas_png_base64` when it is non-empty**.
3. Allow large request bodies (reverse proxy / app server limits).

---

## Backend persistence (this repo’s `POST /api/convert`)

After a successful request, the server writes under `runs/<run_id>/input/`:

| Path | Meaning |
|------|---------|
| `banner.png` | Decoded **`banner_png_base64`**. |
| `raw_figma.json` | **`raw_json`** as saved JSON (includes **`atlas_region`** if the plugin injected it). |
| `elements/atlas.png` | Decoded **`element_atlas_png_base64`** when non-empty. |
| `elements/leaf_NNN.png` | Crops from the atlas (one file per region row, for Qwen / debugging). |
| `elements/element_atlas_crops_manifest.json` | One JSON row per crop (`path`, `atlas_*`, `saved_as`, …). |
| `elements/element_image_refs.json` | **`element_image_refs`** only (metadata; no bitmap per `image_hash`). |

**Qwen / VLM limit:** only the **first 32** cropped leaf PNGs are passed as extra scene images (`element_image_paths`), even if the atlas packs more leaves (the plugin may pack up to 250). Remaining crops exist on disk only.

---

## File reference

Payload assembly: `code.js` (search for `requestBody` near the `/api/convert` `fetch`).
