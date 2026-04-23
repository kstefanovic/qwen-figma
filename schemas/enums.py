from enum import Enum


class ZoneRole(str, Enum):
    TEXT_ZONE = "text_zone"
    IMAGE_ZONE = "image_zone"
    PRODUCT_ZONE = "product_zone"
    LEGAL_ZONE = "legal_zone"
    PROMO_ZONE = "promo_zone"
    BACKGROUND_ZONE = "background_zone"
    OVERLAY_ZONE = "overlay_zone"
    BRAND_ZONE = "brand_zone"


class GroupRole(str, Enum):
    BRAND_GROUP = "brand_group"
    HEADLINE_GROUP = "headline_group"
    PRICE_GROUP = "price_group"
    CTA_GROUP = "cta_group"
    PRODUCT_GROUP = "product_group"
    LEGAL_GROUP = "legal_group"
    BADGE_GROUP = "badge_group"
    DECORATION_GROUP = "decoration_group"
    TEXT_GROUP = "text_group"
    HERO_GROUP = "hero_group"
    BACKGROUND_GROUP = "background_group"
    UNKNOWN = "unknown"


class ElementRole(str, Enum):
    HEADLINE = "headline"
    SUBHEADLINE = "subheadline"
    BODY_TEXT = "body_text"
    LEGAL = "legal"
    PRICE_MAIN = "price_main"
    PRICE_OLD = "price_old"
    PRICE_FRACTION = "price_fraction"
    DISCOUNT_TEXT = "discount_text"
    CTA = "cta"
    LABEL = "label"
    BADGE_TEXT = "badge_text"
    LOGO_TEXT = "logo_text"
    PRODUCT_IMAGE = "product_image"
    HERO_PHOTO = "hero_photo"
    LOGO_ICON = "logo_icon"
    BRAND_MARK = "brand_mark"
    BACKGROUND_SHAPE = "background_shape"
    BACKGROUND_PANEL = "background_panel"
    DECORATION = "decoration"
    DISCOUNT_BADGE = "discount_badge"
    AGE_BADGE = "age_badge"
    PACKSHOT = "packshot"
    TEXT_CONTAINER = "text_container"
    IMAGE_CONTAINER = "image_container"
    PROMO_CONTAINER = "promo_container"
    UNKNOWN = "unknown"


class FunctionalType(str, Enum):
    FUNCTIONAL = "functional"
    DECORATIVE = "decorative"
    BACKGROUND = "background"


class ImportanceLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnchorType(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    LEFT_CENTER = "left_center"
    CENTER = "center"
    RIGHT_CENTER = "right_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    FREE = "free"


class LayoutPattern(str, Enum):
    LEFT_TEXT_RIGHT_IMAGE = "left_text_right_image"
    LEFT_TEXT_RIGHT_PRODUCT = "left_text_right_product"
    PRICE_LEFT_PRODUCT_RIGHT = "price_left_product_right"
    CENTERED_TEXT_DECORATIVE_BACKGROUND = "centered_text_decorative_background"
    FULL_BACKGROUND_TEXT_OVERLAY = "full_background_text_overlay"
    TOP_IMAGE_BOTTOM_TEXT_MOBILE = "top_image_bottom_text_mobile"
    TOP_TEXT_BOTTOM_PRODUCT_MOBILE = "top_text_bottom_product_mobile"
    PRODUCT_DOMINANT_MOBILE = "product_dominant_mobile"
    PROMO_TEXT_ONLY = "promo_text_only"
    CATALOG_PRICE_CARD = "catalog_price_card"
    UNKNOWN = "unknown"


class InternalLayout(str, Enum):
    VERTICAL_STACK = "vertical_stack"
    HORIZONTAL_ROW = "horizontal_row"
    OVERLAY = "overlay"
    FREEFORM = "freeform"
    SINGLE = "single"


class RelationType(str, Enum):
    CONTAINS = "contains"
    BELONGS_TO_ZONE = "belongs_to_zone"
    BELONGS_TO_GROUP = "belongs_to_group"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    OVERLAPS = "overlaps"
    INSIDE = "inside"
    ALIGNED_LEFT = "aligned_left"
    ALIGNED_RIGHT = "aligned_right"
    ALIGNED_CENTER_X = "aligned_center_x"
    ALIGNED_CENTER_Y = "aligned_center_y"
    SAME_ROW = "same_row"
    SAME_COLUMN = "same_column"
    VERTICAL_STACK = "vertical_stack"
    HORIZONTAL_PAIR = "horizontal_pair"
    SAME_GROUP = "same_group"
    SAME_ROLE_FAMILY = "same_role_family"
    CORNER_ANCHOR = "corner_anchor"
    BACKGROUND_FOR = "background_for"
    BADGE_FOR = "badge_for"


class ConstraintType(str, Enum):
    MIN_FONT_SIZE = "min_font_size"
    MAX_CROP_RATIO = "max_crop_ratio"
    MUST_REMAIN_VISIBLE = "must_remain_visible"
    MUST_ANCHOR_CORNER = "must_anchor_corner"
    MUST_PRESERVE_ORDER = "must_preserve_order"
    MUST_STAY_IN_ZONE = "must_stay_in_zone"
    MUST_NOT_OVERLAP = "must_not_overlap"
    MUST_PRESERVE_GROUP = "must_preserve_group"
