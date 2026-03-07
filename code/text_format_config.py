"""
Construction Text Taxonomy & Synthetic Data Generator
====================================================
Defines a two-stage construction taxonomy and domain-specific vocabulary
used to generate synthetic vendor submittal text for classifier training.

Usage:
    from text_format_config import TEXT_TAXONOMY, generate_text_training_data
    df = generate_text_training_data(samples_per_category=100, seed=42)
"""

import random
import itertools
import pandas as pd

# =============================================================================
# TAXONOMY:  Stage-1 & Stage-2 categories
# =============================================================================
TEXT_TAXONOMY = {
    "03": {
        "name": "Concrete",
        "level2": {
            "03 10 00": "Concrete Forming and Accessories",
            "03 20 00": "Concrete Reinforcing",
            "03 30 00": "Cast-in-Place Concrete",
        },
    },
    "05": {
        "name": "Metals",
        "level2": {
            "05 10 00": "Structural Metal Framing",
            "05 30 00": "Metal Decking",
            "05 50 00": "Metal Fabrications",
        },
    },
    "07": {
        "name": "Thermal and Moisture Protection",
        "level2": {
            "07 10 00": "Dampproofing and Waterproofing",
            "07 20 00": "Thermal Insulation",
            "07 50 00": "Membrane Roofing",
        },
    },
    "08": {
        "name": "Openings",
        "level2": {
            "08 10 00": "Doors and Frames",
            "08 50 00": "Windows",
            "08 80 00": "Glazing",
        },
    },
    "09": {
        "name": "Finishes",
        "level2": {
            "09 20 00": "Plaster and Gypsum Board",
            "09 60 00": "Flooring",
            "09 90 00": "Painting and Coating",
        },
    },
    "22": {
        "name": "Plumbing",
        "level2": {
            "22 10 00": "Plumbing Piping and Pumps",
            "22 30 00": "Plumbing Equipment",
            "22 40 00": "Plumbing Fixtures",
        },
    },
    "23": {
        "name": "HVAC",
        "level2": {
            "23 30 00": "HVAC Air Distribution",
            "23 60 00": "Central Cooling Equipment",
            "23 70 00": "Central Heating Equipment",
        },
    },
    "26": {
        "name": "Electrical",
        "level2": {
            "26 05 00": "Common Work Results for Electrical",
            "26 20 00": "Low-Voltage Electrical Distribution",
            "26 50 00": "Lighting",
        },
    },
}

# =============================================================================
# Helper look-ups
# =============================================================================

def get_stage1_labels():
    """Return sorted list of Stage-1 label strings like 'Category 03 - Concrete'."""
    return sorted(
        f"Category {code} - {info['name']}" for code, info in TEXT_TAXONOMY.items()
    )

def get_stage2_labels():
    """Return sorted list of Stage-2 label strings like '03 30 00 Cast-in-Place Concrete'."""
    labels = []
    for stage1_code, stage1_info in TEXT_TAXONOMY.items():
        for stage2_code, stage2_name in stage1_info["level2"].items():
            labels.append(f"{stage2_code} {stage2_name}")
    return sorted(labels)

def stage2_to_stage1(stage2_label: str) -> str:
    """Map a Stage-2 label back to its parent Stage-1 label."""
    stage2_code = stage2_label[:8]  # e.g. '03 30 00'
    stage1_code = stage2_code[:2]   # e.g. '03'
    return f"Category {stage1_code} - {TEXT_TAXONOMY[stage1_code]['name']}"


# Backward-compatible aliases
TAXONOMY = TEXT_TAXONOMY
get_l1_labels = get_stage1_labels
get_l2_labels = get_stage2_labels
l2_to_l1 = stage2_to_stage1

# =============================================================================
# SENTENCE PATTERNS – generic construction document templates
# =============================================================================
SENTENCE_PATTERNS = [
    "Supply and install {item} {context}. {detail} {spec}.",
    "Proposal for {action} of {item} {context}. {detail}",
    "Vendor submittal: {item}. {detail} {spec}.",
    "Scope of work includes {action} of {item} {context}. {detail}",
    "Provide all labor, materials and equipment for {item} {context}. {spec}. {detail}",
    "{action} of {item} including all necessary accessories and hardware. {spec}.",
    "Quotation for {item} {context}. {detail} {spec}.",
    "Furnish and install {item} as indicated on drawings and {spec}. {detail}",
    "Subcontractor bid for {item}. Location: {context}. {detail}",
    "Material submittal for {item}. {detail} Refer to {spec}.",
    "This submittal covers the {action} of {item} {context}. All work to comply with {spec}. {detail}",
    "Request for approval: {item} for use {context}. {detail} {spec}.",
]

# =============================================================================
# CATEGORY-SPECIFIC VOCABULARY  (per Stage-2 category)
# =============================================================================
CATEGORY_VOCAB = {
    # -----------------------------------------------------------------
    # Division 03 – Concrete
    # -----------------------------------------------------------------
    "03 10 00": {
        "items": [
            "plywood concrete formwork system",
            "steel panel forms for walls",
            "aluminum gang forming system",
            "column formwork with snap ties",
            "insulating concrete forms (ICF)",
            "slab edge forms and blockouts",
            "foundation footing formwork",
            "curved radius concrete forms",
            "form release agent and accessories",
            "reshoring and shoring system for elevated slabs",
        ],
        "contexts": [
            "for foundation walls and footings",
            "for cast-in-place concrete columns",
            "for elevated slab construction at level 2 through roof",
            "for retaining wall per structural drawings",
            "for basement foundation walls",
            "at building core shear walls",
        ],
        "details": [
            "Includes form stripping, cleaning, and concrete surface patching.",
            "Snap ties at 24 inches on center both ways.",
            "Forms to remain in place for minimum 7-day cure period.",
            "Plywood grade B-B, minimum 3/4 inch thick.",
            "Includes all bracing, wedges, and form oil.",
            "Allow for three reuses of formwork panels.",
        ],
        "specs": [
            "per ACI 347 standards",
            "per project specification Section 03 10 00",
            "per structural engineer approved shop drawings",
            "conforming to OSHA shoring requirements",
        ],
        "actions": [
            "installation", "supply and erection", "fabrication and installation",
            "furnishing and placement",
        ],
    },
    "03 20 00": {
        "items": [
            "grade 60 deformed reinforcing steel bars",
            "welded wire reinforcement WWR 6x6-W2.9xW2.9",
            "epoxy-coated rebar for exposed concrete",
            "fiber reinforcing mesh for slab-on-grade",
            "post-tensioning tendons and anchors",
            "rebar dowels and starter bars",
            "reinforcing bar couplers (mechanical splices)",
            "stainless steel reinforcement for corrosive environments",
            "bar supports and chairs for rebar placement",
            "headed reinforcing bar anchors",
        ],
        "contexts": [
            "for foundation mat and footings",
            "for elevated concrete slab reinforcement",
            "for concrete columns and beams",
            "for retaining wall and grade beams",
            "for concrete shear walls per seismic design",
            "for parking structure post-tensioned slabs",
        ],
        "details": [
            "Includes all cutting, bending, placing and tying.",
            "Rebar to be shop-fabricated and tagged per bar schedule.",
            "Minimum concrete cover 3 inches for footings, 1.5 inches for slabs.",
            "Lap splices per ACI 318 development length requirements.",
            "Mill certifications required for all reinforcing steel.",
            "Bar supports at maximum 4-foot spacing.",
        ],
        "specs": [
            "per ACI 318-19 and project structural drawings",
            "per ASTM A615 Grade 60 specifications",
            "per CRSI Manual of Standard Practice",
            "in accordance with Section 03 20 00",
        ],
        "actions": [
            "supply and installation", "placement", "furnishing and tying",
            "fabrication and placement",
        ],
    },
    "03 30 00": {
        "items": [
            "4000 PSI ready-mix structural concrete",
            "5000 PSI high-strength concrete for columns",
            "lightweight concrete for elevated slabs",
            "self-consolidating concrete (SCC)",
            "fiber-reinforced concrete (FRC) for slab on grade",
            "air-entrained concrete for exterior flatwork",
            "high-early-strength concrete for fast-track construction",
            "exposed aggregate decorative concrete",
            "concrete placement and finishing for foundations",
            "concrete pump truck and placement services",
        ],
        "contexts": [
            "for building foundations and footings",
            "for elevated concrete slabs at all levels",
            "for concrete columns, beams and shear walls",
            "for slab-on-grade in warehouse area",
            "for exterior sidewalks and curbs",
            "for concrete topping slab over metal deck",
        ],
        "details": [
            "Maximum slump 4 inches without superplasticizer.",
            "Concrete to be cured with wet burlap and plastic sheeting for 7 days.",
            "Includes concrete testing: slump, air content, cylinders per ACI 301.",
            "Portland cement Type I/II with 3/4 inch maximum aggregate.",
            "Water-to-cement ratio not to exceed 0.45.",
            "Finish to be steel trowel for interior slabs, broom finish for exterior.",
        ],
        "specs": [
            "per ACI 318 and ACI 301 specifications",
            "per approved concrete mix design",
            "per project specification Section 03 30 00",
            "conforming to ASTM C94 ready-mixed concrete standard",
        ],
        "actions": [
            "placement and finishing", "supply and placement", "pouring and curing",
            "delivery and installation",
        ],
    },

    # -----------------------------------------------------------------
    # Division 05 – Metals
    # -----------------------------------------------------------------
    "05 10 00": {
        "items": [
            "structural steel wide-flange beams W12x26",
            "steel HSS columns and bracing",
            "steel moment frame connections",
            "structural steel roof trusses",
            "steel angle lintels for masonry openings",
            "steel base plates and anchor bolts",
            "structural steel canopy framing",
            "hot-rolled steel I-beams for floor framing",
            "steel braced frame lateral system",
            "galvanized structural steel for exterior",
        ],
        "contexts": [
            "for main building structural frame",
            "for second floor and roof framing",
            "for steel canopy at building entrance",
            "for mezzanine floor framing",
            "for loading dock steel structure",
            "for lateral bracing system per seismic design",
        ],
        "details": [
            "Material per ASTM A992 Grade 50 for wide flanges.",
            "All connections per AISC standards and structural drawings.",
            "Includes shop drawings, erection drawings and mill certificates.",
            "Steel to be blast-cleaned and primed with shop primer.",
            "High-strength bolts A325 for all connections.",
            "Includes crane and rigging for steel erection.",
        ],
        "specs": [
            "per AISC 360 Specification for Structural Steel Buildings",
            "per project specification Section 05 10 00",
            "conforming to AWS D1.1 for welded connections",
            "per structural engineer sealed shop drawings",
        ],
        "actions": [
            "fabrication and erection", "supply and installation", "furnishing and erection",
            "shop fabrication and field erection",
        ],
    },
    "05 30 00": {
        "items": [
            "composite metal floor deck 3-inch 20 gauge",
            "metal roof deck 1.5-inch Type B 22 gauge",
            "galvanized corrugated steel decking",
            "acoustic metal deck with perforations",
            "cellular metal floor deck for electrical raceway",
            "metal deck pour stops and closures",
            "metal deck shear studs and connectors",
            "non-composite roof deck panels",
        ],
        "contexts": [
            "for composite floor system at all elevated levels",
            "for roof deck installation over steel joists",
            "for mezzanine floor deck",
            "for parking garage composite deck system",
            "for metal building roof deck replacement",
        ],
        "details": [
            "Deck to be welded to supports at 12 inches on center.",
            "Includes all side laps, end laps, and pour stop angles.",
            "Minimum bearing length 2 inches on structural steel supports.",
            "Deck to be Vulcraft or approved equal.",
            "Includes temporary shoring during concrete placement.",
        ],
        "specs": [
            "per SDI (Steel Deck Institute) standards",
            "per project specification Section 05 30 00",
            "per composite design per AISC 360 Chapter I",
            "conforming to ASTM A653 galvanized steel",
        ],
        "actions": [
            "supply and installation", "furnishing and welding",
            "delivery and installation", "erection",
        ],
    },
    "05 50 00": {
        "items": [
            "miscellaneous steel angles, plates and channels",
            "steel pipe bollards with concrete fill",
            "steel stair stringers, treads and handrails",
            "steel ladder and safety cage",
            "ornamental steel railing and balusters",
            "steel embed plates and loose lintels",
            "galvanized steel shelf angles for masonry",
            "steel dumpster enclosure frame",
            "equipment support steel framing",
            "steel access hatches and floor plates",
        ],
        "contexts": [
            "at stairwells, mechanical rooms and roof",
            "for building perimeter bollard protection",
            "for interior and exterior metal railings",
            "miscellaneous metals throughout building",
            "for equipment pads and roof penetration supports",
        ],
        "details": [
            "All exposed steel to be hot-dip galvanized after fabrication.",
            "Handrails to comply with ADA requirements.",
            "Includes all anchors, fasteners and connection hardware.",
            "Steel to be shop-primed with rust-inhibitive primer.",
            "Field welds to be ground smooth and touch-up painted.",
        ],
        "specs": [
            "per project specification Section 05 50 00",
            "conforming to ASTM A36 for miscellaneous steel",
            "per ADA and local building code requirements",
            "per architectural detail drawings",
        ],
        "actions": [
            "fabrication and installation", "supply and erection",
            "shop fabrication and field installation",
        ],
    },

    # -----------------------------------------------------------------
    # Division 07 – Thermal and Moisture Protection
    # -----------------------------------------------------------------
    "07 10 00": {
        "items": [
            "below-grade bituminous waterproofing membrane",
            "fluid-applied rubberized asphalt waterproofing",
            "bentonite clay waterproofing panels",
            "crystalline waterproofing admixture for concrete",
            "cementitious dampproofing coating",
            "sheet membrane waterproofing for plaza deck",
            "foundation drain board and protection course",
            "liquid-applied air and vapor barrier",
            "traffic-bearing waterproofing for parking deck",
            "underslab vapor barrier 15-mil polyethylene",
        ],
        "contexts": [
            "for below-grade foundation walls and footings",
            "for occupied basement and parking levels",
            "for plaza deck over occupied space",
            "for elevator pit and mechanical room",
            "at tunnel and underground utility connections",
        ],
        "details": [
            "Membrane to be installed on clean, dry substrate.",
            "Includes protection board and drainage composite.",
            "All seams to be sealed and corners reinforced.",
            "10-year manufacturer warranty on waterproofing system.",
            "Surface preparation includes primer coat as recommended by manufacturer.",
        ],
        "specs": [
            "per ASTM D6153 for liquid-applied waterproofing",
            "per project specification Section 07 10 00",
            "per manufacturer installation guidelines",
            "conforming to local building code moisture protection requirements",
        ],
        "actions": [
            "application", "installation", "supply and application",
            "furnishing and applying",
        ],
    },
    "07 20 00": {
        "items": [
            "fiberglass batt insulation R-19 for exterior walls",
            "rigid XPS foam insulation board 2-inch thick",
            "spray-applied polyurethane foam insulation (SPF)",
            "mineral wool semi-rigid insulation board",
            "blown-in cellulose insulation for attic spaces",
            "polyiso rigid insulation for roof assembly",
            "continuous exterior insulation (CI) system",
            "pipe and duct insulation with vapor jacket",
            "acoustic insulation for interior partitions",
            "foam board insulation for slab edge",
        ],
        "contexts": [
            "for exterior wall cavities and stud bays",
            "for roof insulation above deck",
            "for mechanical room pipe and duct insulation",
            "for attic floor and knee wall insulation",
            "for continuous insulation at building envelope",
        ],
        "details": [
            "Insulation to be installed without gaps, voids or compression.",
            "Includes vapor retarder on warm side of assembly.",
            "R-value verified by third-party testing per ASTM C518.",
            "Fire rating: Class A flame spread per ASTM E84.",
            "Insulation to be unfaced or kraft-faced as indicated.",
        ],
        "specs": [
            "per ASHRAE 90.1 energy code requirements",
            "per project specification Section 07 20 00",
            "per manufacturer technical data and installation guide",
            "conforming to local energy conservation code",
        ],
        "actions": [
            "installation", "supply and installation", "furnishing and installing",
            "application",
        ],
    },
    "07 50 00": {
        "items": [
            "TPO single-ply membrane roofing system 60-mil",
            "EPDM rubber roofing membrane fully adhered",
            "PVC single-ply roofing membrane with hot-air welded seams",
            "modified bitumen SBS roofing system two-ply",
            "built-up roofing (BUR) 4-ply asphalt system",
            "standing seam metal roof panels 24-gauge",
            "vegetated green roof assembly with drainage mat",
            "roofing insulation tapered cricket system",
            "roof walkway pads and protection mats",
            "roofing penetration flashings and pitch pockets",
        ],
        "contexts": [
            "for low-slope roof area approximately 45,000 square feet",
            "for roof replacement over existing building",
            "for new construction main roof and canopy areas",
            "for rooftop mechanical equipment area",
            "for green roof system over parking structure",
        ],
        "details": [
            "System includes 20-year NDL (No Dollar Limit) manufacturer warranty.",
            "Membrane to be mechanically fastened or fully adhered per wind uplift.",
            "Includes all flashings, counterflashings, and edge metal.",
            "Insulation R-30 minimum per energy code, tapered for drainage.",
            "Remove existing roofing to deck and install new system complete.",
        ],
        "specs": [
            "per NRCA roofing guidelines",
            "per project specification Section 07 50 00",
            "per FM Global and UL listed roof assembly",
            "per manufacturer approved applicator installation manual",
        ],
        "actions": [
            "installation", "supply and installation", "tear-off and replacement",
            "furnishing and applying",
        ],
    },

    # -----------------------------------------------------------------
    # Division 08 – Openings
    # -----------------------------------------------------------------
    "08 10 00": {
        "items": [
            "hollow metal door frames 16-gauge welded",
            "solid core wood doors with hardware",
            "fire-rated steel doors and frames 90-minute",
            "FRP fiberglass reinforced plastic doors",
            "aluminum storefront entry doors",
            "overhead coiling steel service doors",
            "acoustic sound-rated door assemblies STC 45",
            "access doors and panels flush-mounted",
            "automatic sliding door operator",
            "steel security doors with electronic hardware",
        ],
        "contexts": [
            "for interior and exterior door openings per door schedule",
            "for fire-rated corridor and stairwell openings",
            "for mechanical and electrical room access",
            "for main building entry and vestibule",
            "for warehouse and loading dock openings",
        ],
        "details": [
            "Hardware to include lever handles, closers, and hinges per hardware schedule.",
            "Fire-rated assemblies with UL listed label.",
            "Frames to be set and grouted plumb and square.",
            "Finish: factory-applied powder coat paint.",
            "Includes all weatherstripping, thresholds and smoke seals.",
        ],
        "specs": [
            "per project specification Section 08 10 00",
            "per NFPA 80 fire door assembly standards",
            "per door and hardware schedule on drawings",
            "conforming to SDI (Steel Door Institute) standards",
        ],
        "actions": [
            "supply and installation", "furnishing and hanging",
            "delivery and installation", "procurement and installation",
        ],
    },
    "08 50 00": {
        "items": [
            "aluminum double-hung windows thermally broken",
            "vinyl casement windows with Low-E glass",
            "fixed aluminum picture windows",
            "aluminum awning windows with screens",
            "impact-resistant hurricane windows",
            "aluminum sliding windows",
            "steel fixed-lite windows for industrial",
            "curtain wall window system unitized",
            "operable skylight window units",
            "blast-resistant security window assemblies",
        ],
        "contexts": [
            "for exterior building facade per window schedule",
            "for office and classroom window openings",
            "for residential tower window replacement",
            "for hospital patient room exterior windows",
            "for high-rise building perimeter",
        ],
        "details": [
            "Glazing to be dual-pane insulated Low-E argon-filled.",
            "U-factor 0.30 maximum, SHGC 0.25 maximum per energy code.",
            "Includes all sealants, flashing and perimeter caulking.",
            "Windows to be factory-assembled and glazed.",
            "Color: clear anodized aluminum or custom RAL color.",
        ],
        "specs": [
            "per AAMA/WDMA/CSA 101 performance standards",
            "per project specification Section 08 50 00",
            "per energy code and ASHRAE 90.1 requirements",
            "per approved window schedule and shop drawings",
        ],
        "actions": [
            "supply and installation", "furnishing and glazing",
            "procurement and installation", "delivery and setting",
        ],
    },
    "08 80 00": {
        "items": [
            "1-inch insulated glazing units (IGU) Low-E coated",
            "tempered safety glass panels",
            "laminated security glass for storefronts",
            "spandrel glass panels with ceramic frit",
            "structural silicone glazed curtain wall panels",
            "fire-rated glazing assemblies 60-minute",
            "back-painted decorative glass panels",
            "triple-glazed high-performance IGU",
            "wired glass for fire-rated openings",
            "bird-friendly fritted glass panels",
        ],
        "contexts": [
            "for curtain wall and storefront glazing systems",
            "for interior glass partitions and sidelights",
            "for building lobby and atrium glazing",
            "for canopy and skylight overhead glazing",
            "for fire-rated glazing at rated corridors",
        ],
        "details": [
            "Glass to meet ANSI Z97.1 safety glazing requirements.",
            "Insulated units with warm-edge spacer and argon gas fill.",
            "Includes structural silicone sealant for wet glazing.",
            "Glass thickness and type per structural analysis for wind loads.",
            "Mock-up panel required for architect approval before production.",
        ],
        "specs": [
            "per ASTM C1048 for flat glass specifications",
            "per project specification Section 08 80 00",
            "per IGCC standards for insulating glass",
            "per glazing contractor shop drawings",
        ],
        "actions": [
            "supply and installation", "glazing", "furnishing and setting",
            "procurement and installation",
        ],
    },

    # -----------------------------------------------------------------
    # Division 09 – Finishes
    # -----------------------------------------------------------------
    "09 20 00": {
        "items": [
            "5/8-inch Type X gypsum board for fire-rated partitions",
            "metal stud framing 3-5/8 inch 20-gauge",
            "moisture-resistant (green board) gypsum in wet areas",
            "sound-rated gypsum board assembly STC 50",
            "shaft wall liner and C-H stud system",
            "glass mat gypsum sheathing for exterior",
            "drywall finishing including taping, mudding and sanding",
            "curved gypsum board soffits and bulkheads",
            "abuse-resistant gypsum board for corridors",
            "veneer plaster system over blueboard",
        ],
        "contexts": [
            "for interior partitions and ceilings throughout building",
            "for fire-rated corridor and stairwell assemblies",
            "for bathroom and kitchen wet-area walls",
            "for tenant improvement build-out",
            "for elevator shaft and mechanical shaft walls",
        ],
        "details": [
            "Partitions to achieve fire rating per UL assembly design.",
            "Taping and finishing to Level 4 finish per ASTM C840.",
            "Includes acoustic sealant at top and bottom tracks.",
            "Metal studs at 16 inches on center, bridging at mid-height.",
            "Corner bead at all exposed edges and reveals.",
        ],
        "specs": [
            "per GA (Gypsum Association) Fire Resistance Design Manual",
            "per project specification Section 09 20 00",
            "per ASTM C840 gypsum board application standard",
            "per UL listed assembly design number",
        ],
        "actions": [
            "installation", "supply and installation", "framing and boarding",
            "furnishing and finishing",
        ],
    },
    "09 60 00": {
        "items": [
            "porcelain tile flooring 24x24 inch with thin-set mortar",
            "luxury vinyl plank (LVP) flooring click-lock system",
            "epoxy resin flooring for commercial kitchen",
            "polished concrete floor with densifier and sealer",
            "carpet tile modular 24x24 with adhesive",
            "hardwood oak strip flooring 3/4-inch solid",
            "rubber athletic flooring for gymnasium",
            "terrazzo flooring with brass divider strips",
            "raised access flooring system 6-inch pedestal",
            "sheet vinyl flooring for healthcare corridors",
        ],
        "contexts": [
            "for lobby, corridors and common areas",
            "for office spaces and conference rooms",
            "for commercial kitchen and food prep areas",
            "for restrooms and locker rooms",
            "for gymnasium and multipurpose room",
        ],
        "details": [
            "Includes all floor preparation, leveling and patching.",
            "Transition strips at all flooring material changes.",
            "Adhesive to be low-VOC per LEED IEQ requirements.",
            "Moisture testing required per ASTM F2170 before installation.",
            "Base: 4-inch rubber or tile cove base at all walls.",
        ],
        "specs": [
            "per TCNA Handbook for tile installations",
            "per project specification Section 09 60 00",
            "per manufacturer installation instructions",
            "per ASTM F710 for substrate preparation",
        ],
        "actions": [
            "supply and installation", "furnishing and laying",
            "installation", "procurement and installation",
        ],
    },
    "09 90 00": {
        "items": [
            "interior latex paint (2 coats) over primer",
            "exterior acrylic paint system with primer",
            "high-performance epoxy wall coating for wet areas",
            "intumescent fireproofing paint for structural steel",
            "anti-graffiti coating for exterior masonry",
            "wood stain and polyurethane finish for millwork",
            "elastomeric exterior wall coating",
            "zinc-rich primer for corrosion protection",
            "floor sealer and concrete stain",
            "specialty antimicrobial paint for healthcare",
        ],
        "contexts": [
            "for all interior walls and ceilings",
            "for exterior building facade and soffits",
            "for mechanical and electrical rooms",
            "for exposed structural steel fireproofing",
            "for millwork, casework and trim",
        ],
        "details": [
            "Two coats finish over one coat primer on new drywall.",
            "Colors per architect-selected paint schedule.",
            "Includes surface preparation, sanding and priming.",
            "VOC content to comply with LEED IEQ v4 requirements.",
            "Touch-up painting at all trades damage at project completion.",
        ],
        "specs": [
            "per MPI (Master Painters Institute) standards",
            "per project specification Section 09 90 00",
            "per paint manufacturer technical data sheets",
            "per SSPC surface preparation standards for steel",
        ],
        "actions": [
            "application", "supply and application", "painting",
            "surface preparation and coating",
        ],
    },

    # -----------------------------------------------------------------
    # Division 22 – Plumbing
    # -----------------------------------------------------------------
    "22 10 00": {
        "items": [
            "copper Type L water supply piping and fittings",
            "PVC DWV drain, waste and vent piping",
            "cast iron no-hub soil pipe and fittings",
            "CPVC hot and cold water distribution piping",
            "PEX flexible water supply tubing",
            "domestic water booster pump station",
            "sump pump and pit with float switch",
            "stainless steel piping for medical gas",
            "insulated chilled water piping for plumbing",
            "backflow preventer and pressure reducing valve",
        ],
        "contexts": [
            "for building domestic water distribution system",
            "for sanitary sewer and storm drain piping",
            "for mechanical room plumbing connections",
            "for restroom and kitchen plumbing rough-in",
            "for roof drain piping and overflow system",
        ],
        "details": [
            "Includes all valves, supports, hangers and seismic bracing.",
            "Piping to be pressure tested at 1.5x operating pressure.",
            "Insulation on all hot and cold water piping per specification.",
            "Pipe connections: soldered for copper, solvent welded for PVC.",
            "Includes core drilling, firestopping and sleeve installation.",
        ],
        "specs": [
            "per Uniform Plumbing Code (UPC)",
            "per project specification Section 22 10 00",
            "per ASME B31.9 building services piping",
            "per local plumbing code and health department requirements",
        ],
        "actions": [
            "installation", "supply and installation", "rough-in and connection",
            "furnishing and piping",
        ],
    },
    "22 30 00": {
        "items": [
            "commercial gas-fired water heater 100-gallon",
            "tankless instantaneous water heater",
            "water softener and treatment system",
            "grease interceptor for commercial kitchen",
            "electric water cooler / drinking fountain",
            "sewage ejector pump with alarm",
            "recirculating hot water pump system",
            "thermostatic mixing valve station",
            "expansion tank for domestic water system",
            "point-of-use water filtration system",
        ],
        "contexts": [
            "for domestic hot water system in mechanical room",
            "for kitchen grease waste management",
            "for building water treatment and softening",
            "for below-grade sewage ejection system",
            "for drinking water cooling and dispensing",
        ],
        "details": [
            "Equipment includes mounting, piping connections and startup.",
            "Manufacturer factory startup and commissioning required.",
            "Includes electrical connections by others coordination required.",
            "5-year manufacturer warranty on heat exchanger.",
            "Unit to be seismic-rated for installation zone.",
        ],
        "specs": [
            "per ASME Boiler and Pressure Vessel Code",
            "per project specification Section 22 30 00",
            "per NSF/ANSI 61 for drinking water components",
            "per local health and plumbing code requirements",
        ],
        "actions": [
            "supply and installation", "procurement and installation",
            "furnishing and connecting", "delivery and startup",
        ],
    },
    "22 40 00": {
        "items": [
            "vitreous china water closet (toilet) floor-mounted",
            "wall-hung lavatory sink with faucet",
            "stainless steel commercial kitchen sink triple-bowl",
            "ADA-compliant accessible plumbing fixtures",
            "automatic sensor faucets and flush valves",
            "mop basin (service sink) floor-mounted",
            "wall-mounted urinal with flush valve",
            "emergency eyewash and shower station",
            "janitor closet plumbing fixtures",
            "drinking fountain / bottle filler combo unit",
        ],
        "contexts": [
            "for public restrooms per fixture schedule",
            "for commercial kitchen and break rooms",
            "for ADA-compliant accessible restroom",
            "for janitor closets and service areas",
            "for laboratory and medical exam rooms",
        ],
        "details": [
            "All fixtures to be water-efficient per WaterSense label.",
            "Includes all trim, supplies, stops and connection fittings.",
            "Chrome-plated brass faucets with ceramic disc cartridge.",
            "Fixtures to be white vitreous china unless noted otherwise.",
            "ADA fixtures mounted at heights per accessibility code.",
        ],
        "specs": [
            "per ASME A112.19.2 for vitreous china fixtures",
            "per project specification Section 22 40 00",
            "per ADA and ICC/ANSI A117.1 accessibility standard",
            "per plumbing fixture schedule on drawings",
        ],
        "actions": [
            "supply and installation", "furnishing and setting",
            "procurement and mounting", "delivery and installation",
        ],
    },

    # -----------------------------------------------------------------
    # Division 23 – HVAC
    # -----------------------------------------------------------------
    "23 30 00": {
        "items": [
            "galvanized sheet metal ductwork rectangular and round",
            "flexible insulated duct for branch connections",
            "VAV (variable air volume) terminal units with controls",
            "supply air diffusers and return air grilles",
            "duct-mounted fire and smoke dampers",
            "fabric ductwork for open ceiling areas",
            "ductwork insulation wrap and lining",
            "kitchen exhaust hood and duct system",
            "outside air louvers with bird screen",
            "duct silencers and sound attenuators",
        ],
        "contexts": [
            "for HVAC air distribution throughout building",
            "for rooftop air handling unit connections",
            "for commercial kitchen ventilation system",
            "for office and conference room supply and return",
            "for data center precision air distribution",
        ],
        "details": [
            "Ductwork fabricated per SMACNA standards.",
            "Includes all hangers, supports and seismic bracing per code.",
            "Duct leakage testing per SMACNA Class A seal requirement.",
            "Insulation: 2-inch fiberglass duct wrap with vapor barrier.",
            "TAB (Testing Adjusting Balancing) included in scope.",
        ],
        "specs": [
            "per SMACNA HVAC Duct Construction Standards",
            "per project specification Section 23 30 00",
            "per ASHRAE 90.1 duct insulation requirements",
            "per mechanical drawings and duct layout schedule",
        ],
        "actions": [
            "fabrication and installation", "supply and installation",
            "furnishing and connecting", "duct installation",
        ],
    },
    "23 60 00": {
        "items": [
            "packaged rooftop air conditioning unit 25-ton",
            "air-cooled scroll chiller 100-ton capacity",
            "water-cooled centrifugal chiller 500-ton",
            "split-system DX cooling unit with condenser",
            "computer room precision cooling unit (CRAC/CRAH)",
            "cooling tower induced draft crossflow",
            "VRF (variable refrigerant flow) outdoor condensing units",
            "chilled water air handler unit (AHU) with economizer",
            "evaporative condenser for refrigeration",
            "mini-split ductless air conditioning system",
        ],
        "contexts": [
            "for building central cooling plant",
            "for rooftop packaged HVAC installation",
            "for data center and server room cooling",
            "for office building comfort cooling",
            "for tenant space supplemental cooling",
        ],
        "details": [
            "Unit to be high-efficiency with minimum 14 SEER rating.",
            "Includes refrigerant piping, pad, and electrical connections.",
            "Factory startup and commissioning by manufacturer representative.",
            "Vibration isolation pads and curb adapter included.",
            "Controls: BACnet compatible DDC for building automation integration.",
        ],
        "specs": [
            "per ASHRAE 90.1 minimum efficiency requirements",
            "per project specification Section 23 60 00",
            "per AHRI certified performance ratings",
            "per mechanical schedule on drawings",
        ],
        "actions": [
            "supply and installation", "procurement and installation",
            "delivery, rigging and startup", "furnishing and commissioning",
        ],
    },
    "23 70 00": {
        "items": [
            "gas-fired hot water boiler 2000 MBH",
            "electric resistance duct heater",
            "condensing high-efficiency boiler 95% AFUE",
            "hydronic baseboard fin-tube radiation heater",
            "unit heater gas-fired for warehouse",
            "radiant ceiling panel heating system",
            "heat recovery ventilator (HRV/ERV)",
            "steam-to-hot-water heat exchanger",
            "infrared radiant tube heater for garage",
            "hot water circulating pump and piping",
        ],
        "contexts": [
            "for building central heating system",
            "for mechanical room boiler plant",
            "for warehouse and loading dock space heating",
            "for perimeter heating at exterior walls",
            "for ventilation air preheat system",
        ],
        "details": [
            "Boiler efficiency exceeds ASHRAE 90.1 minimum requirement.",
            "Includes gas piping, venting and combustion air provisions.",
            "Factory startup required by manufacturer certified technician.",
            "Controls integrated with building automation system (BAS).",
            "Expansion tank, air separator and chemical feeder included.",
        ],
        "specs": [
            "per ASME Boiler and Pressure Vessel Code for boilers",
            "per project specification Section 23 70 00",
            "per AHRI certified heating capacity ratings",
            "per local mechanical and fuel gas code",
        ],
        "actions": [
            "supply and installation", "procurement and installation",
            "delivery and commissioning", "furnishing and piping",
        ],
    },

    # -----------------------------------------------------------------
    # Division 26 – Electrical
    # -----------------------------------------------------------------
    "26 05 00": {
        "items": [
            "EMT and rigid metal conduit with fittings",
            "copper building wire THHN/THWN various sizes",
            "cable tray system ladder type with covers",
            "electrical junction boxes, pull boxes and fittings",
            "grounding and bonding system with ground rods",
            "firestopping for electrical penetrations",
            "wire connectors, lugs and termination kits",
            "underground PVC electrical conduit duct bank",
            "seismic bracing for electrical equipment",
            "electrical identification labels and tags",
        ],
        "contexts": [
            "for electrical raceway and wiring infrastructure",
            "for power distribution conduit and wire pulls",
            "for data center cable tray installation",
            "for underground electrical duct bank to building",
            "for grounding electrode system per NEC",
        ],
        "details": [
            "All wiring and raceway per National Electrical Code (NEC).",
            "Conduit fill not to exceed 40% per NEC Chapter 9.",
            "Includes all supports, straps and seismic bracing.",
            "Wire pulled with approved lubricant, no damage to insulation.",
            "As-built drawings required showing all concealed raceway routing.",
        ],
        "specs": [
            "per NEC (NFPA 70) National Electrical Code",
            "per project specification Section 26 05 00",
            "per UL listed materials and components",
            "per NECA installation standards",
        ],
        "actions": [
            "installation", "supply and installation", "furnishing and pulling",
            "rough-in and termination",
        ],
    },
    "26 20 00": {
        "items": [
            "main electrical switchboard 2000A 480/277V",
            "distribution panelboard 225A 208/120V",
            "automatic transfer switch (ATS) 400A",
            "dry-type transformer 75 kVA 480V to 208/120V",
            "diesel emergency standby generator 500 kW",
            "UPS (uninterruptible power supply) 100 kVA",
            "motor control center (MCC) with VFDs",
            "bus duct feeder system from switchboard",
            "surge protective device (SPD) panel-mounted",
            "electrical metering and monitoring system",
        ],
        "contexts": [
            "for main electrical service and distribution",
            "for emergency and standby power system",
            "for electrical room equipment installation",
            "for building power distribution infrastructure",
            "for tenant electrical metering and sub-distribution",
        ],
        "details": [
            "Equipment to be UL 891 listed and labeled.",
            "Includes circuit breakers, fuses and all internal components.",
            "Arc flash study and labels required per NFPA 70E.",
            "Generator includes sub-base fuel tank, muffler and weatherproof enclosure.",
            "Load bank testing and startup by factory technician.",
        ],
        "specs": [
            "per NEC Article 230 (Services) and 240 (Overcurrent Protection)",
            "per project specification Section 26 20 00",
            "per NFPA 110 for emergency power systems",
            "per utility company service entrance requirements",
        ],
        "actions": [
            "supply and installation", "procurement and installation",
            "furnishing and connecting", "delivery and energization",
        ],
    },
    "26 50 00": {
        "items": [
            "LED recessed troffer light fixtures 2x4",
            "LED high-bay warehouse light fixtures",
            "exterior LED wall pack and pole-mounted fixtures",
            "architectural pendant lighting for lobby",
            "emergency and exit lighting with battery backup",
            "dimming lighting control system with occupancy sensors",
            "LED strip lighting for cove and accent",
            "explosion-proof lighting for hazardous locations",
            "parking garage LED fixtures with daylight sensor",
            "sports field LED flood lighting system",
        ],
        "contexts": [
            "for interior lighting per reflected ceiling plan",
            "for exterior site and building-mounted lighting",
            "for warehouse and industrial high-bay areas",
            "for office, lobby and common area lighting",
            "for parking garage and canopy lighting",
        ],
        "details": [
            "All fixtures to be LED with minimum 50,000-hour rated life.",
            "Includes dimming drivers compatible with 0-10V control system.",
            "Emergency fixtures with 90-minute battery backup per code.",
            "Color temperature: 4000K for offices, 3000K for corridors.",
            "Lighting controls to include occupancy sensors per energy code.",
        ],
        "specs": [
            "per NEC Article 410 and local energy code",
            "per project specification Section 26 50 00",
            "per IES recommended lighting levels for occupancy type",
            "per DesignLights Consortium (DLC) qualified products list",
        ],
        "actions": [
            "supply and installation", "furnishing and connecting",
            "procurement and mounting", "delivery and wiring",
        ],
    },
}

# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_text_training_data(samples_per_category: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic vendor submittal documents for two-stage text classification.

    Parameters
    ----------
    samples_per_category : int
        Number of text samples to generate for each Stage-2 category.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
        full_item, stage1_code, stage1_name, stage2_code, stage2_name,
        stage1_label, stage2_label
    """
    rng = random.Random(seed)
    records = []

    for stage1_code, stage1_info in TEXT_TAXONOMY.items():
        stage1_label = f"Category {stage1_code} - {stage1_info['name']}"

        for stage2_code, stage2_name in stage1_info["level2"].items():
            stage2_label = f"{stage2_code} {stage2_name}"
            vocab = CATEGORY_VOCAB[stage2_code]

            for _ in range(samples_per_category):
                pattern = rng.choice(SENTENCE_PATTERNS)
                text = pattern.format(
                    item=rng.choice(vocab["items"]),
                    context=rng.choice(vocab["contexts"]),
                    detail=rng.choice(vocab["details"]),
                    spec=rng.choice(vocab["specs"]),
                    action=rng.choice(vocab["actions"]),
                )
                # Optionally prepend a random document-level header
                if rng.random() < 0.3:
                    header = rng.choice([
                        f"RE: {stage2_name} –",
                        "VENDOR SUBMITTAL:",
                        "Subject: Material Submittal –",
                        "PROJECT PROPOSAL:",
                        "SCOPE OF WORK DESCRIPTION:",
                        "BID ITEM:",
                    ])
                    text = f"{header} {text}"

                records.append({
                    "full_item": text,
                    "stage1_code": stage1_code,
                    "stage1_name": stage1_info["name"],
                    "stage2_code": stage2_code,
                    "stage2_name": stage2_name,
                    "stage1_label": stage1_label,
                    "stage2_label": stage2_label,
                })

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# Backward-compatible alias
generate_training_data = generate_text_training_data


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    df = generate_text_training_data(samples_per_category=5, seed=0)
    print(f"Generated {len(df)} samples across {df['stage2_label'].nunique()} Stage-2 categories.")
    print(f"Stage-1 distribution:\n{df['stage1_label'].value_counts()}")
    print(f"\nSample:\n{df.iloc[0]['full_item']}")
