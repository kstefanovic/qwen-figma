/**
 * Figma plugin main thread. HTTP contract for ``POST …/api/convert`` is documented in
 * ``API_CONVERT_CONTRACT.md``. Raster payload is **two** Base64 PNGs only: full banner +
 * one element atlas (no per-hash image library).
 *
 * ``POST …/api/v2/analyze-text-zone-visual-json`` (JSON ``banner_png_base64``, same style as ``/api/convert``):
 * orientation, zone_type, and ``text_zone.groups`` (brand_group, headline_group, optional age_badge, legal_text + normalized bboxes) — banner only; no atlas/raw tree.
 */
figma.showUI(__html__, { width: 400, height: 580 });

function normalizeType(type) {
  return String(type || "").toLowerCase().replace(/_/g, " ");
}

function getOrigin(node) {
  const t = node.absoluteTransform;
  return { x: t[0][2], y: t[1][2] };
}

function absoluteBox(node, origin) {
  const t = node.absoluteTransform;
  return {
    x: Number((t[0][2] - origin.x).toFixed(2)),
    y: Number((t[1][2] - origin.y).toFixed(2)),
    width: Number(node.width.toFixed(2)),
    height: Number(node.height.toFixed(2))
  };
}

function serializeNode(node, origin, path) {
  const base = {
    id: node.id,
    path: path,
    name: node.name,
    type: normalizeType(node.type),
    bounds: absoluteBox(node, origin),
    visible: node.visible !== false,
    opacity: typeof node.opacity === "number" ? Number(node.opacity.toFixed(3)) : 1
  };

  if ("characters" in node) {
    base.characters = node.characters;
    if ("fontSize" in node) base.fontSize = node.fontSize;
    if ("fontName" in node) base.fontName = node.fontName;
    if ("textAlignHorizontal" in node) base.textAlignHorizontal = node.textAlignHorizontal;
    if ("textAlignVertical" in node) base.textAlignVertical = node.textAlignVertical;
  }

  if ("layoutMode" in node) {
    base.layoutMode = node.layoutMode;
    base.itemSpacing = node.itemSpacing;
    base.padding = {
      top: node.paddingTop,
      right: node.paddingRight,
      bottom: node.paddingBottom,
      left: node.paddingLeft
    };
  }

  if ("children" in node && Array.isArray(node.children)) {
    base.children = node.children.map((child, index) => {
      const childPath = path === "" ? String(index) : `${path}/${index}`;
      return serializeNode(child, origin, childPath);
    });
  }

  return base;
}

async function exportFramePngBytes(node) {
  return await node.exportAsync({
    format: "PNG",
    constraint: { type: "SCALE", value: 1 }
  });
}

/** Max leaves packed into the element atlas. Keep in sync with backend MAX_ATLAS_REGIONS. */
const MAX_ELEMENT_LAYER_PNGS = 512;

/** Space between element bounding boxes. Requested as 20x the previous 12px spacing. */
const PREVIOUS_ATLAS_GAP = 12;
const ATLAS_GAP = PREVIOUS_ATLAS_GAP * 20;
const ATLAS_CELL_PADDING = Math.round(ATLAS_GAP / 2);
const ATLAS_MAX_ROW_WIDTH = 8192;
const ATLAS_MAX_CELL = 4096;
const ELEMENTS_PNG_MAX_WIDTH = 1920;
const ELEMENTS_PNG_MAX_HEIGHT = 1028;
const ATLAS_BOX_COLOR = { r: 1, g: 0, b: 0 };
const ATLAS_CELL_COLOR = { r: 1, g: 1, b: 1 };

/**
 * Collect **leaf** scene nodes (same rules as former per-PNG export): no children
 * (except INSTANCE as a single leaf), visible, min size 1px.
 */
function collectLeafElementRefs(root, maxCount) {
  const out = [];
  let count = 0;

  function visit(node, path) {
    if (count >= maxCount) return;
    if (node.visible === false) return;
    if (!("width" in node) || node.width < 1 || node.height < 1) return;

    const kids =
      "children" in node && Array.isArray(node.children) ? node.children : [];
    const hasKids = kids.length > 0;

    if (hasKids && node.type !== "INSTANCE") {
      for (let i = 0; i < kids.length; i++) {
        const childPath = path === "" ? String(i) : `${path}/${i}`;
        visit(kids[i], childPath);
        if (count >= maxCount) return;
      }
      return;
    }

    out.push({ path, node });
    count++;
  }

  const top = root.children;
  if (!top || !top.length) return out;
  for (let i = 0; i < top.length; i++) {
    visit(top[i], String(i));
    if (count >= maxCount) break;
  }
  if (count >= maxCount) {
    console.warn("collectLeafElementRefs: hit backend atlas-region cap", maxCount);
  }
  return out;
}

function atlasExportScale(width, height) {
  const w = Math.max(1, Number(width) || 1);
  const h = Math.max(1, Number(height) || 1);
  return Math.min(1, ELEMENTS_PNG_MAX_WIDTH / w, ELEMENTS_PNG_MAX_HEIGHT / h);
}

function scaledRegionValue(value, scale) {
  return Math.max(0, Math.round((Number(value) || 0) * scale));
}

function scaledRegionSize(value, scale) {
  return Math.max(1, Math.round((Number(value) || 0) * scale));
}

function makeBoundingBoxRect(x, y, width, height, exportScale) {
  const rect = figma.createRectangle();
  rect.name = "__element_bbox__";
  rect.x = x;
  rect.y = y;
  rect.resizeWithoutConstraints(Math.max(1, width), Math.max(1, height));
  rect.fills = [];
  rect.strokes = [{ type: "SOLID", color: ATLAS_BOX_COLOR }];
  rect.strokeWeight = Math.max(4, 6 / Math.max(exportScale, 0.01));
  rect.strokeAlign = "INSIDE";
  return rect;
}

function makeAtlasCellFrame(x, y, width, height) {
  const cell = figma.createFrame();
  cell.name = "__element_cell__";
  cell.x = x;
  cell.y = y;
  cell.layoutMode = "NONE";
  cell.clipsContent = true;
  cell.fills = [
    {
      type: "SOLID",
      color: ATLAS_CELL_COLOR,
      opacity: 0.04,
    },
  ];
  cell.resizeWithoutConstraints(Math.max(1, width), Math.max(1, height));
  return cell;
}

/**
 * Clone leaves into one off-screen frame, pack in rows with large spacing, draw visible
 * bounding boxes, and export a **single** PNG atlas capped to 1920 x 1028.
 * Returns Base64 PNG + region list in final exported pixel coords. Names/paths match ``raw_json``.
 */
async function buildElementAtlasPngAndRegions(root, maxCount) {
  const entries = collectLeafElementRefs(root, maxCount);
  if (!entries.length) {
    return {
      atlasPngBase64: "",
      regions: [],
      atlasSize: { width: 0, height: 0, source_width: 0, source_height: 0, scale: 1 },
    };
  }

  const atlas = figma.createFrame();
  atlas.name = "__plugin_element_atlas__";
  atlas.fills = [];
  atlas.layoutMode = "NONE";
  atlas.clipsContent = false;
  figma.currentPage.appendChild(atlas);
  atlas.x = -120000;
  atlas.y = -120000;

  const layoutRegions = [];
  const bboxRects = [];
  let curX = 0;
  let curY = 0;
  let rowH = 0;

  try {
    for (const { path, node } of entries) {
      let clone;
      try {
        clone = node.clone();
      } catch (e) {
        console.warn("buildElementAtlas: clone failed", path, e);
        continue;
      }

      try {
        if ("resizeWithoutConstraints" in clone && typeof clone.resizeWithoutConstraints === "function") {
          const tw = Math.min(clone.width, ATLAS_MAX_CELL);
          const th = Math.min(clone.height, ATLAS_MAX_CELL);
          if (tw < clone.width || th < clone.height) {
            clone.resizeWithoutConstraints(tw, th);
          }
        }
      } catch (e) {
        /* keep natural size */
      }

      const cw = clone.width;
      const ch = clone.height;
      const cellW = cw + ATLAS_CELL_PADDING * 2;
      const cellH = ch + ATLAS_CELL_PADDING * 2;

      if (curX + cellW + ATLAS_GAP > ATLAS_MAX_ROW_WIDTH && curX > 0) {
        curY += rowH + ATLAS_GAP;
        curX = 0;
        rowH = 0;
      }

      const cell = makeAtlasCellFrame(curX, curY, cellW, cellH);
      atlas.appendChild(cell);
      cell.appendChild(clone);
      clone.x = ATLAS_CELL_PADDING;
      clone.y = ATLAS_CELL_PADDING;
      const bboxRect = makeBoundingBoxRect(ATLAS_CELL_PADDING, ATLAS_CELL_PADDING, cw, ch, 1);
      cell.appendChild(bboxRect);
      bboxRects.push(bboxRect);

      layoutRegions.push({
        path,
        node_id: node.id,
        name: node.name,
        type: normalizeType(node.type),
        atlas_x: Math.round(curX + ATLAS_CELL_PADDING),
        atlas_y: Math.round(curY + ATLAS_CELL_PADDING),
        atlas_width: Math.round(cw),
        atlas_height: Math.round(ch),
        cell_x: Math.round(curX),
        cell_y: Math.round(curY),
        cell_width: Math.round(cellW),
        cell_height: Math.round(cellH),
      });

      curX += cellW + ATLAS_GAP;
      rowH = Math.max(rowH, cellH);
    }

    let maxR = 0;
    let maxB = 0;
    for (const region of layoutRegions) {
      maxR = Math.max(maxR, region.cell_x + region.cell_width);
      maxB = Math.max(maxB, region.cell_y + region.cell_height);
    }
    const finalW = Math.max(1, Math.ceil(maxR));
    const finalH = Math.max(1, Math.ceil(maxB));
    const scale = atlasExportScale(finalW, finalH);
    for (const bboxRect of bboxRects) {
      bboxRect.strokeWeight = Math.max(4, 6 / Math.max(scale, 0.01));
    }

    if ("resizeWithoutConstraints" in atlas) {
      atlas.resizeWithoutConstraints(finalW, finalH);
    }

    const bytes = await atlas.exportAsync({
      format: "PNG",
      constraint: { type: "SCALE", value: scale },
    });
    const regions = layoutRegions.map((region) =>
      Object.assign({}, region, {
        atlas_x: scaledRegionValue(region.atlas_x, scale),
        atlas_y: scaledRegionValue(region.atlas_y, scale),
        atlas_width: scaledRegionSize(region.atlas_width, scale),
        atlas_height: scaledRegionSize(region.atlas_height, scale),
        atlas_cell_x: scaledRegionValue(region.cell_x, scale),
        atlas_cell_y: scaledRegionValue(region.cell_y, scale),
        atlas_cell_width: scaledRegionSize(region.cell_width, scale),
        atlas_cell_height: scaledRegionSize(region.cell_height, scale),
        atlas_scale: Number(scale.toFixed(6)),
      }),
    );
    return {
      atlasPngBase64: uint8ToBase64(bytes),
      regions,
      atlasSize: {
        width: scaledRegionSize(finalW, scale),
        height: scaledRegionSize(finalH, scale),
        source_width: finalW,
        source_height: finalH,
        scale: Number(scale.toFixed(6)),
      },
    };
  } finally {
    atlas.remove();
  }
}

/**
 * Add ``atlas_region: { x, y, width, height }`` on each ``raw_json`` node whose ``path``
 * appears in the atlas (same ``path`` / ``name`` as serialization).
 */
function injectAtlasRegionsIntoRawJson(rawJson, regions) {
  if (!rawJson || !Array.isArray(regions) || regions.length === 0) return;
  const byPath = new Map(
    regions.map((r) => [
      r.path,
      {
        x: r.atlas_x,
        y: r.atlas_y,
        width: r.atlas_width,
        height: r.atlas_height,
      },
    ]),
  );

  function walk(n) {
    if (!n || typeof n !== "object") return;
    if (typeof n.path === "string" && byPath.has(n.path)) {
      n.atlas_region = byPath.get(n.path);
    }
    if (Array.isArray(n.children)) {
      n.children.forEach(walk);
    }
  }

  walk(rawJson);
}

function attachAtlasMetadataToRawJson(rawJson, atlasSize, regions) {
  if (!rawJson || typeof rawJson !== "object") return;
  rawJson.element_atlas = {
    file_name: "elements.png",
    max_width: ELEMENTS_PNG_MAX_WIDTH,
    max_height: ELEMENTS_PNG_MAX_HEIGHT,
    width: atlasSize && atlasSize.width ? atlasSize.width : 0,
    height: atlasSize && atlasSize.height ? atlasSize.height : 0,
    source_width: atlasSize && atlasSize.source_width ? atlasSize.source_width : 0,
    source_height: atlasSize && atlasSize.source_height ? atlasSize.source_height : 0,
    scale: atlasSize && atlasSize.scale ? atlasSize.scale : 1,
    region_count: Array.isArray(regions) ? regions.length : 0,
    bbox_gap_px: ATLAS_GAP,
    bbox_style: "red stroke inside each element region",
  };
}

function uint8ToBase64(bytes) {
  const base64abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let result = "";
  let i;

  for (i = 0; i + 2 < bytes.length; i += 3) {
    result += base64abc[bytes[i] >> 2];
    result += base64abc[((bytes[i] & 0x03) << 4) | (bytes[i + 1] >> 4)];
    result += base64abc[((bytes[i + 1] & 0x0f) << 2) | (bytes[i + 2] >> 6)];
    result += base64abc[bytes[i + 2] & 0x3f];
  }

  if (i < bytes.length) {
    result += base64abc[bytes[i] >> 2];

    if (i === bytes.length - 1) {
      result += base64abc[(bytes[i] & 0x03) << 4];
      result += "==";
    } else {
      result += base64abc[((bytes[i] & 0x03) << 4) | (bytes[i + 1] >> 4)];
      result += base64abc[(bytes[i + 1] & 0x0f) << 2];
      result += "=";
    }
  }

  return result;
}

function stampOriginalNodeIds(root) {
  let stamped = 0;

  function walk(node) {
    try {
      node.setPluginData("originalNodeId", node.id);
      stamped++;
    } catch (e) {
      console.warn("Failed stamping node:", node && node.id, e);
    }

    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) {
        walk(child);
      }
    }
  }

  walk(root);
  return stamped;
}

function collectClonedNodesByOriginalId(root) {
  const map = new Map();
  let mapped = 0;

  function walk(node) {
    try {
      const originalId = node.getPluginData("originalNodeId");
      if (originalId) {
        map.set(originalId, node);
        mapped++;
      }
    } catch (e) {
      console.warn("Failed collecting cloned node map entry:", node && node.id, e);
    }

    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) {
        walk(child);
      }
    }
  }

  walk(root);
  return { map, mapped };
}

function asArray(value) {
  return Array.isArray(value) ? value.filter(Boolean) : [];
}

function getNodeByPath(root, path) {
  if (!root || typeof path !== "string") return null;
  const trimmed = path.trim();
  if (!trimmed) return root;

  const segments = trimmed.split("/");
  let current = root;

  for (const segment of segments) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0) return null;
    if (!("children" in current) || !Array.isArray(current.children)) return null;
    if (index >= current.children.length) return null;
    current = current.children[index];
  }

  return current;
}

function getTopLevelNodeByPath(root, path) {
  if (!root || typeof path !== "string") return null;
  const trimmed = path.trim();
  if (!trimmed) return null;

  const first = trimmed.split("/")[0];
  const index = Number(first);
  if (!Number.isInteger(index) || index < 0) return null;
  if (!("children" in root) || !Array.isArray(root.children)) return null;
  if (index >= root.children.length) return null;
  return root.children[index];
}

function getSemanticName(item) {
  if (!item) return null;
  if (typeof item === "string") {
    const value = item.trim();
    return value || null;
  }
  if (typeof item === "object") {
    return item.semantic_name || item.semanticName || item.role || null;
  }
  return null;
}

function sanitizeLayerName(name) {
  return String(name)
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_а-яА-ЯёЁ:-]/g, "");
}

function setSemanticName(node, itemOrName) {
  const rawName = typeof itemOrName === "string" ? itemOrName : getSemanticName(itemOrName);
  if (!node || !rawName) return false;

  const clean = sanitizeLayerName(rawName);
  if (!clean) return false;

  node.name = clean;
  node.setPluginData("semanticName", clean);

  if (typeof itemOrName !== "string" && itemOrName && typeof itemOrName === "object") {
    if (itemOrName.role) node.setPluginData("semanticRole", String(itemOrName.role));
    if (itemOrName.source_figma_id) node.setPluginData("sourceFigmaId", String(itemOrName.source_figma_id));
    if (itemOrName.figma_node_id) node.setPluginData("sourceFigmaId", String(itemOrName.figma_node_id));
    if (itemOrName.confidence !== undefined) {
      node.setPluginData("semanticConfidence", String(itemOrName.confidence));
    }
  }

  return true;
}

function cloneFrameBeside(sourceFrame) {
  const convertedFrame = sourceFrame.clone();
  convertedFrame.x = sourceFrame.x + sourceFrame.width + 80;
  convertedFrame.y = sourceFrame.y;
  convertedFrame.name = sourceFrame.name;
  figma.currentPage.appendChild(convertedFrame);
  return convertedFrame;
}

function buildPathNodeMap(root) {
  const map = new Map();
  map.set("", root);

  function walk(node, path) {
    if (!("children" in node) || !Array.isArray(node.children)) return;
    node.children.forEach((child, index) => {
      const childPath = path ? `${path}/${index}` : String(index);
      map.set(childPath, child);
      walk(child, childPath);
    });
  }

  walk(root, "");
  return map;
}

function getAncestorUnderRoot(root, node) {
  if (!root || !node) return null;
  let current = node;
  while (current && current.parent && current.parent.id !== root.id) {
    current = current.parent;
  }
  return current && current.parent && current.parent.id === root.id ? current : null;
}

function deriveContainerSemanticName(itemOrName) {
  const base = String(getSemanticName(itemOrName) || itemOrName || "").trim().toLowerCase();
  if (!base) return null;
  if (base === "headline") return "headline_group";
  if (base === "legal_text" || base === "legal") return "legal_group";
  if (
    base === "brand_name_yandex" ||
    base === "brand_name_lavka" ||
    base === "logo" ||
    base === "logo_heart" ||
    base === "logo_ellipse"
  ) {
    return "brand_group";
  }
  if (base === "age_badge") return "badge_group";
  if (base.indexOf("product_") === 0 || base.indexOf("hero_") === 0) return "hero_group";
  if (base === "decoration_star" || /^decoration_star(_\d+)?$/.test(base)) {
    return "decoration_star_group";
  }
  if (base.indexOf("decoration_") === 0) return "decoration_group";
  if (base.indexOf("background_") === 0) return "background_group";
  return null;
}

function isGenericLayerName(name) {
  const n = String(name || "").trim();
  if (!n) return true;
  if (/^\d+$/.test(n)) return true;
  if (/^(group|rectangle|vector|ellipse|line|polygon|star|frame|text)\s+\d+$/i.test(n)) return true;
  return false;
}

function collectGenericNodes(root) {
  const generic = [];
  function walk(node) {
    if (isGenericLayerName(node.name)) {
      generic.push({ id: node.id, name: node.name });
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children) walk(child);
    }
  }
  walk(root);
  return generic;
}

function resolveNodeByBackendIdOrPath(item, nodeByOriginalId, pathNodeMap) {
  if (!item) return { node: null, matchedBy: "" };
  const id = String(item.source_figma_id || item.figma_node_id || item.node_id || item.id || "").trim();
  const path = String(item.path || "").trim();
  if (id && nodeByOriginalId.has(id)) return { node: nodeByOriginalId.get(id), matchedBy: "id" };
  if (path && pathNodeMap.has(path)) return { node: pathNodeMap.get(path), matchedBy: "path" };
  return { node: null, matchedBy: "" };
}

function canSafelyGroup(children) {
  if (!Array.isArray(children) || children.length < 2) return false;
  const parent = children[0].parent;
  if (!parent) return false;
  if (!children.every((child) => child.parent && child.parent.id === parent.id)) return false;
  if (parent.type === "INSTANCE" || parent.type === "COMPONENT" || parent.type === "COMPONENT_SET") return false;
  if (children.some((child) => "isMask" in child && child.isMask)) return false;
  return true;
}

function applySemanticFallbackName(node, groupName) {
  const base = String(groupName || "").toLowerCase();
  let fallback = "visual_asset";
  if (base.indexOf("logo") !== -1 || base.indexOf("brand") !== -1) fallback = "brand_text_part";
  else if (base.indexOf("decoration") !== -1) fallback = "decoration_part";
  else if (base.indexOf("background") !== -1) fallback = "background_part";
  else if (base.indexOf("hero") !== -1 || base.indexOf("product") !== -1) fallback = "logo_part";
  return setSemanticName(node, fallback);
}

function applyUpdatesWithVisibleContainers(convertedFrame, backendResponse, nodeByOriginalId, pathNodeMap) {
  const updates = asArray(backendResponse && backendResponse.updates);
  const sourceUpdateMap = new Map();
  let updatesAppliedById = 0;
  let updatesAppliedByPath = 0;
  let renamedVisibleContainers = 0;
  const missingSourceIds = new Set();
  const missingPaths = new Set();
  const explicitlyNamedNodeIds = new Set();

  function renameContainerForUpdate(update, exactNode) {
    const path = String(update.path || "").trim();
    const parentSemanticName = String(update.parent_semantic_name || "").trim();
    const containerName = parentSemanticName || deriveContainerSemanticName(update);
    if (!containerName) return;

    let containerNode = null;
    if (path) {
      const parentPath = path.indexOf("/") > -1 ? path.split("/").slice(0, -1).join("/") : "";
      containerNode = parentPath ? pathNodeMap.get(parentPath) : getTopLevelNodeByPath(convertedFrame, path);
    }
    if (!containerNode && exactNode && exactNode.parent) {
      containerNode = exactNode.parent.id === convertedFrame.id ? getAncestorUnderRoot(convertedFrame, exactNode) : exactNode.parent;
    }
    if (!containerNode || (exactNode && containerNode.id === exactNode.id)) return;

    if (setSemanticName(containerNode, containerName)) {
      renamedVisibleContainers++;
      explicitlyNamedNodeIds.add(containerNode.id);
    }
  }

  for (const update of updates) {
    const sourceId = String(update && update.source_figma_id ? update.source_figma_id : "").trim();
    const path = String(update && update.path ? update.path : "").trim();
    if (sourceId) sourceUpdateMap.set(sourceId, update);

    if (!getSemanticName(update)) {
      if (sourceId) missingSourceIds.add(sourceId);
      if (path) missingPaths.add(path);
      continue;
    }

    const resolved = resolveNodeByBackendIdOrPath(update, nodeByOriginalId, pathNodeMap);
    if (resolved.node && resolved.node.id !== convertedFrame.id) {
      if (setSemanticName(resolved.node, update)) {
        explicitlyNamedNodeIds.add(resolved.node.id);
        if (resolved.matchedBy === "id") updatesAppliedById++;
        else updatesAppliedByPath++;
      }
      renameContainerForUpdate(update, resolved.node);
    } else {
      if (sourceId) missingSourceIds.add(sourceId);
      if (path) missingPaths.add(path);
    }
  }

  return {
    updates,
    sourceUpdateMap,
    updatesAppliedById,
    updatesAppliedByPath,
    renamedVisibleContainers,
    missingSourceIds,
    missingPaths,
    explicitlyNamedNodeIds
  };
}

function applySemanticsToClone(result, convertedFrame) {
  const preservedRootName = convertedFrame.name;
  const { map: nodeByOriginalId, mapped } = collectClonedNodesByOriginalId(convertedFrame);
  const pathNodeMap = buildPathNodeMap(convertedFrame);
  const updatesSummary = applyUpdatesWithVisibleContainers(convertedFrame, result, nodeByOriginalId, pathNodeMap);
  const semanticElements = asArray(result && result.semantic && result.semantic.elements);
  const semanticGroups = asArray(result && result.semantic && result.semantic.groups);
  const elementById = new Map();

  semanticElements.forEach((el) => {
    const id = String(
      el && (el.figma_node_id || el.source_figma_id || el.node_id || el.id)
        ? (el.figma_node_id || el.source_figma_id || el.node_id || el.id)
        : ""
    ).trim();
    if (id) elementById.set(id, el);
  });

  let semanticElementsApplied = 0;
  let semanticGroupsApplied = 0;
  let groupsCreated = 0;

  for (const element of semanticElements) {
    const id = String(
      element && (element.figma_node_id || element.source_figma_id || element.node_id || element.id)
        ? (element.figma_node_id || element.source_figma_id || element.node_id || element.id)
        : ""
    ).trim();

    const resolved = resolveNodeByBackendIdOrPath(element, nodeByOriginalId, pathNodeMap);
    if (resolved.node && resolved.node.id !== convertedFrame.id && setSemanticName(resolved.node, element)) {
      semanticElementsApplied++;
      updatesSummary.explicitlyNamedNodeIds.add(resolved.node.id);
    } else {
      if (id) updatesSummary.missingSourceIds.add(id);
      if (element && element.path) updatesSummary.missingPaths.add(String(element.path));
    }
  }

  for (const group of semanticGroups) {
    const groupName = getSemanticName(group);
    const children = asArray(group && group.children);
    if (!groupName || children.length === 0) continue;

    let groupApplied = false;
    const knownContainerId = String(
      group && (group.figma_node_id || group.source_figma_id || group.node_id || group.source_node_id)
        ? (group.figma_node_id || group.source_figma_id || group.node_id || group.source_node_id)
        : ""
    ).trim();
    if (knownContainerId && nodeByOriginalId.has(knownContainerId)) {
      const containerNode = nodeByOriginalId.get(knownContainerId);
      if (containerNode.id !== convertedFrame.id && setSemanticName(containerNode, group)) {
        updatesSummary.renamedVisibleContainers++;
        updatesSummary.explicitlyNamedNodeIds.add(containerNode.id);
        groupApplied = true;
      }
    }

    const matchedChildren = [];
    const parentMap = new Map();
    for (const childRaw of children) {
      const childItem = typeof childRaw === "object" ? childRaw : { source_figma_id: childRaw };
      const childId = String(
        childItem && (childItem.source_figma_id || childItem.figma_node_id || childItem.node_id || childItem.id)
          ? (childItem.source_figma_id || childItem.figma_node_id || childItem.node_id || childItem.id)
          : ""
      ).trim();

      const resolvedChild = resolveNodeByBackendIdOrPath(childItem, nodeByOriginalId, pathNodeMap);
      const childNode = resolvedChild.node;
      if (!childNode) {
        if (childId) updatesSummary.missingSourceIds.add(childId);
        continue;
      }

      matchedChildren.push(childNode);
      if (childNode.parent) {
        parentMap.set(childNode.parent.id, childNode.parent);
      }

      const childElement = childId ? elementById.get(childId) : null;
      const childUpdate = childId ? updatesSummary.sourceUpdateMap.get(childId) : null;

      if (childElement && getSemanticName(childElement)) {
        setSemanticName(childNode, childElement);
        updatesSummary.explicitlyNamedNodeIds.add(childNode.id);
      } else if (childUpdate && getSemanticName(childUpdate)) {
        setSemanticName(childNode, childUpdate);
        updatesSummary.explicitlyNamedNodeIds.add(childNode.id);
      } else if (isGenericLayerName(childNode.name)) {
        applySemanticFallbackName(childNode, groupName);
      }

      const childPath = String(childItem && childItem.path ? childItem.path : "").trim();
      if (childPath) {
        const topLevelNode = getTopLevelNodeByPath(convertedFrame, childPath);
        if (topLevelNode && setSemanticName(topLevelNode, group)) {
          updatesSummary.renamedVisibleContainers++;
          updatesSummary.explicitlyNamedNodeIds.add(topLevelNode.id);
          groupApplied = true;
        }
      }
    }

    if (canSafelyGroup(matchedChildren)) {
      try {
        const groupNode = figma.group(matchedChildren, matchedChildren[0].parent);
        if (setSemanticName(groupNode, group)) {
          groupsCreated++;
          groupApplied = true;
        }
      } catch (e) {
        console.warn("Failed creating semantic group:", e);
      }
    } else if (parentMap.size === 1) {
      const existingContainer = Array.from(parentMap.values())[0];
      if (existingContainer && existingContainer.id !== convertedFrame.id && setSemanticName(existingContainer, group)) {
        updatesSummary.renamedVisibleContainers++;
        updatesSummary.explicitlyNamedNodeIds.add(existingContainer.id);
        groupApplied = true;
      }
    }

    if (groupApplied) semanticGroupsApplied++;
  }

  const remainingGeneric = collectGenericNodes(convertedFrame);
  convertedFrame.name = preservedRootName;

  return {
    mapped,
    updatesReceived: updatesSummary.updates.length,
    updatesAppliedById: updatesSummary.updatesAppliedById,
    updatesAppliedByPath: updatesSummary.updatesAppliedByPath,
    semanticElementsReceived: semanticElements.length,
    semanticElementsApplied,
    semanticGroupsReceived: semanticGroups.length,
    semanticGroupsApplied,
    renamedVisibleContainers: updatesSummary.renamedVisibleContainers,
    groupsCreated,
    groupsSkipped: Math.max(0, semanticGroups.length - semanticGroupsApplied),
    missingSourceIds: Array.from(updatesSummary.missingSourceIds),
    missingPaths: Array.from(updatesSummary.missingPaths),
    remainingGeneric
  };
}

function getSelectionInfo() {
  const selection = figma.currentPage.selection;

  if (selection.length === 0) {
    return { hasSelection: false };
  }

  const node = selection[0];

  return {
    hasSelection: true,
    id: node.id,
    name: node.name,
    type: node.type,
    isFrame: node.type === "FRAME",
    width: "width" in node ? Number(node.width.toFixed(2)) : null,
    height: "height" in node ? Number(node.height.toFixed(2)) : null
  };
}

function sendSelectionInfo() {
  figma.ui.postMessage({
    type: "selection-info",
    selection: getSelectionInfo()
  });
}

figma.on("selectionchange", () => {
  sendSelectionInfo();
});

function postStatus(message) {
  figma.ui.postMessage({ type: "status", message });
}

function postError(message) {
  figma.ui.postMessage({ type: "error", message });
}

figma.ui.onmessage = async (msg) => {
  if (msg.type === "classify-zone-selected-frame") {
    const selection = figma.currentPage.selection;
    if (selection.length !== 1 || selection[0].type !== "FRAME") {
      postError("Select exactly one frame.");
      sendSelectionInfo();
      return;
    }
    const selectedFrame = selection[0];
    const backendUrl = String(msg.backendUrl || "").trim().replace(/\/+$/, "");
    if (!backendUrl) {
      postError("Backend URL is empty.");
      return;
    }
    const requestUrl = backendUrl + "/api/v2/analyze-text-zone-visual-json";
    try {
      const origin = getOrigin(selectedFrame);
      const rawJson = serializeNode(selectedFrame, origin, "");
      rawJson.templateId = "figma_plugin";
      postStatus("Exporting banner.png…");
      const pngBytes = await exportFramePngBytes(selectedFrame);
      const pngBase64 = uint8ToBase64(pngBytes);
      postStatus("Calling visual zone + text-zone analysis with raw JSON…");
      console.log("POST analyze-text-zone-visual:", requestUrl);
      const requestBody = { banner_png_base64: pngBase64, raw_json: rawJson };
      const response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      const text = await response.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch (parseErr) {
        data = null;
      }
      if (!response.ok) {
        const detail =
          data && typeof data === "object" && data.detail != null
            ? typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail)
            : text || "HTTP " + String(response.status);
        throw new Error(detail);
      }
      if (!data || typeof data !== "object") {
        throw new Error("Invalid JSON from backend.");
      }
      postStatus("Zone + text-zone analysis complete.");
      figma.ui.postMessage({
        type: "zone-classify-result",
        ok: true,
        result: data,
      });
      figma.notify(
        "Zone: " +
          data.zone_type +
          " (" +
          Number(data.confidence).toFixed(2) +
          ")",
        { timeout: 5 },
      );
      figma.ui.postMessage({ type: "done" });
      sendSelectionInfo();
    } catch (err) {
      console.error("Analyze text-zone visual failed:", err);
      var msgText =
        err && err.stack
          ? err.message + "\n\n" + err.stack
          : String(err && err.message ? err.message : err);
      if (err && err.message === "Failed to fetch") {
        msgText +=
          "\n\nFigma only allows requests to origins listed in manifest.json " +
          "networkAccess.devAllowedDomains (scheme, host, and port must match exactly). " +
          "Example: http://localhost:30079 and http://127.0.0.1:30079 are different. " +
          "After changing manifest.json, reload the plugin from Plugins → Development.";
      }
      postError(msgText);
      figma.ui.postMessage({ type: "zone-classify-result", ok: false });
      sendSelectionInfo();
    }
    return;
  }

  if (msg.type !== "convert-selected-frame") return;

  const selection = figma.currentPage.selection;

  if (selection.length !== 1 || selection[0].type !== "FRAME") {
    postError("Select exactly one frame.");
    sendSelectionInfo();
    return;
  }

  const selectedFrame = selection[0];
  const origin = getOrigin(selectedFrame);

  try {
    console.log("Selected frame:", {
      id: selectedFrame.id,
      name: selectedFrame.name
    });

    const stampedNodeCount = stampOriginalNodeIds(selectedFrame);
    console.log("Number of original nodes stamped:", stampedNodeCount);

    postStatus("Step 1/5: Serializing selected frame...");

    let rawJson = serializeNode(selectedFrame, origin, "");
    rawJson.templateId = "figma_plugin";

    postStatus("Step 2/5: Exporting banner.png...");
    const pngBytes = await exportFramePngBytes(selectedFrame);

    postStatus("Step 3/5: Encoding banner.png...");
    const pngBase64 = uint8ToBase64(pngBytes);

    postStatus("Step 4/5: Building elements.png with visible bounding boxes…");
    const {
      atlasPngBase64: elementAtlasPngBase64,
      regions: elementAtlasRegions,
      atlasSize: elementAtlasSize,
    } =
      await buildElementAtlasPngAndRegions(selectedFrame, MAX_ELEMENT_LAYER_PNGS);
    injectAtlasRegionsIntoRawJson(rawJson, elementAtlasRegions);
    attachAtlasMetadataToRawJson(rawJson, elementAtlasSize, elementAtlasRegions);
    console.log("Element atlas regions:", elementAtlasRegions.length);
    console.log("elements.png size:", elementAtlasSize);

    let targetWidth = selectedFrame.width;
    let targetHeight = selectedFrame.height;

    if (msg.targetPreset && msg.targetPreset !== "same") {
      const [w, h] = msg.targetPreset.split("x").map(Number);
      targetWidth = w;
      targetHeight = h;
    }

    const requestUrl = `${msg.backendUrl}/api/convert`;
    console.log("Calling backend:", requestUrl);

    postStatus("Step 5/5: Calling backend...");

    const requestedMode = String(msg.convertMode || "apply_to_clone_fast").trim();
    const useQwen = requestedMode === "apply_to_clone_vlm";
    const qwenMode = requestedMode === "apply_to_clone_vlm" ? "scene_only" : undefined;

    let response;

    try {
      // Backend ``ConvertRequest`` uses ``extra='allow'`` so forward-compatible keys are kept.
      const requestBody = {
        banner_png_base64: pngBase64,
        raw_json: rawJson,
        target_width: targetWidth,
        target_height: targetHeight,
        mode: requestedMode || "apply_to_clone_fast",
        use_qwen: useQwen,
        element_atlas_png_base64: elementAtlasPngBase64,
        element_atlas_regions: elementAtlasRegions,
        element_atlas_regions_count: elementAtlasRegions.length,
      };
      if (qwenMode) {
        requestBody.qwen_mode = qwenMode;
      }

      let bodyString;
      try {
        bodyString = JSON.stringify(requestBody);
      } catch (stringifyErr) {
        console.error("JSON.stringify(requestBody) failed:", stringifyErr);
        throw new Error(
          `Failed to serialize request (try fewer layers or smaller frame): ${
            stringifyErr && stringifyErr.message ? stringifyErr.message : stringifyErr
          }`,
        );
      }

      const approxMb = (bodyString.length / (1024 * 1024)).toFixed(2);
      console.log(
        "POST /api/convert payload:",
        bodyString.length,
        "chars (~",
        approxMb,
        "MB), atlas regions:",
        elementAtlasRegions.length,
      );
      postStatus(
        `Sending ~${approxMb} MB (banner.png + elements.png + raw JSON, ${elementAtlasRegions.length} boxes). Backend will flatten raw JSON to mid.json…`,
      );

      response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: bodyString,
      });
    } catch (e) {
      console.error("Fetch failed before response:", e);
      throw new Error(`Fetch failed for ${requestUrl}: ${e && e.message ? e.message : e}`);
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Backend error ${response.status}: ${text}`);
    }

    const result = await response.json();
    console.log("Backend result:", result);
    if (result.debug && result.debug.mid_json_path) {
      console.log("Backend mid.json:", result.debug.mid_json_path);
    }

    if (!result || typeof result !== "object") {
      throw new Error("Backend response is invalid.");
    }

    const supportedModes = new Set([
      "apply_to_clone",
      "apply_to_clone_fast",
      "apply_to_clone_vlm",
      "full_layout_debug"
    ]);
    if (result.mode && !supportedModes.has(result.mode)) {
      throw new Error(`Unsupported backend mode: ${result.mode}`);
    }

    postStatus("Applying semantics to cloned frame...");
    const convertedFrame = cloneFrameBeside(selectedFrame);
    console.log("Converted frame:", {
      id: convertedFrame.id,
      name: convertedFrame.name
    });

    const semanticSummary = applySemanticsToClone(result, convertedFrame);
    console.log("Total cloned nodes:", semanticSummary.mapped);
    console.log("Updates received:", semanticSummary.updatesReceived);
    console.log("Updates applied by id:", semanticSummary.updatesAppliedById);
    console.log("Updates applied by path:", semanticSummary.updatesAppliedByPath);
    console.log("Semantic elements received:", semanticSummary.semanticElementsReceived);
    console.log("Semantic elements applied:", semanticSummary.semanticElementsApplied);
    console.log("Semantic groups received:", semanticSummary.semanticGroupsReceived);
    console.log("Semantic groups applied:", semanticSummary.semanticGroupsApplied);
    if (result.debug && result.debug.mid_json_path) {
      console.log("Mid JSON used by backend:", result.debug.mid_json_path);
    }
    console.log("Renamed visible containers:", semanticSummary.renamedVisibleContainers);
    console.log("Groups created:", semanticSummary.groupsCreated);
    console.log("Groups skipped:", semanticSummary.groupsSkipped);
    console.log("Unmatched backend ids:", semanticSummary.missingSourceIds);
    console.log("Unmatched backend paths:", semanticSummary.missingPaths);
    console.log("Nodes still generic names:", semanticSummary.remainingGeneric);

    figma.currentPage.selection = [convertedFrame];
    figma.viewport.scrollAndZoomIntoView([selectedFrame, convertedFrame]);

    figma.notify(
      `Converted frame created with semantic naming.\n` +
        `Stamped nodes: ${stampedNodeCount}\n` +
        `Mapped cloned nodes: ${semanticSummary.mapped}\n` +
        `Updates: ${semanticSummary.updatesReceived}\n` +
        `Updates by id/path: ${semanticSummary.updatesAppliedById}/${semanticSummary.updatesAppliedByPath}\n` +
        `Semantic elements: ${semanticSummary.semanticElementsReceived}\n` +
        `Semantic elements applied: ${semanticSummary.semanticElementsApplied}\n` +
        `Semantic groups: ${semanticSummary.semanticGroupsReceived}\n` +
        `Semantic groups applied: ${semanticSummary.semanticGroupsApplied}\n` +
        `Renamed visible containers: ${semanticSummary.renamedVisibleContainers}\n` +
        `Groups created: ${semanticSummary.groupsCreated}\n` +
        `Groups skipped: ${semanticSummary.groupsSkipped}\n` +
        `Missing source ids: ${semanticSummary.missingSourceIds.length}\n` +
        `Missing paths: ${semanticSummary.missingPaths.length}\n` +
        `Generic names remaining: ${semanticSummary.remainingGeneric.length}`,
      { timeout: 8 }
    );

    figma.ui.postMessage({ type: "done" });
    sendSelectionInfo();
  } catch (err) {
    console.error("Convert failed:", err);
    postError(
      err && err.stack
        ? `${err.message}\n\n${err.stack}`
        : String(err && err.message ? err.message : err)
    );
  }
};

sendSelectionInfo();