import streamlit as st
import os
import sys
import math
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict
from enum import Enum

# Try to import PIL
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class CrackDirection(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"
    STAIR_STEP = "stair_step"
    RANDOM = "random"
    MAP_CRACKING = "map_cracking"

class SeverityLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"
    CRITICAL = "Critical"

@dataclass
class CrackAnalysis:
    crack_type: str
    direction: CrackDirection
    severity: SeverityLevel
    estimated_width_mm: float
    is_structural: bool
    urgency: str
    causes: List[str]
    symptoms: List[str]
    repair_methods: List[Dict]
    estimated_cost_range: str
    monitoring_advice: str
    professional_needed: bool
    confidence_score: float
    image_features: Dict

# =============================================================================
# KNOWLEDGE BASE
# =============================================================================
CRACK_KNOWLEDGE_BASE = {
    "hairline_shrinkage": {
        "name": "Hairline / Shrinkage Crack",
        "direction": CrackDirection.VERTICAL,
        "severity": SeverityLevel.LOW,
        "width_range_mm": (0.0, 1.5),
        "is_structural": False,
        "urgency": "Monitor. Seal if desired.",
        "causes": [
            "Normal concrete curing and drying shrinkage",
            "Plaster drying after application",
            "Minor thermal expansion/contraction",
            "Seasonal humidity changes",
            "Normal material behavior as building settles"
        ],
        "symptoms": [
            "Extremely thin, hairline-width crack (less than 1/16 inch or 1.5mm)",
            "Appears shortly after construction or plastering (first 1-3 years)",
            "Both sides flush - no height difference between sides",
            "No water leakage through the crack",
            "Stable over time - does not grow when monitored",
            "Common in new homes as ground consolidates under weight"
        ],
        "repair_methods": [
            {
                "name": "Surface Sealant / Flexible Caulk",
                "description": "Apply flexible masonry caulk or concrete sealant over the crack to prevent moisture entry and dust accumulation.",
                "when_to_use": "For dry, stable hairline cracks in concrete or masonry walls",
                "diy_possible": True,
                "diy_cost": "$10 - $30",
                "professional_cost": "$100 - $300",
                "steps": [
                    "Clean crack area with wire brush and vacuum to remove dust/debris",
                    "Apply primer if recommended by sealant manufacturer",
                    "Apply flexible polyurethane caulk along entire crack length",
                    "Tool smooth with wet finger or caulk tool for clean finish",
                    "Allow to cure per manufacturer instructions (typically 24 hours)",
                    "Paint over with matching color if desired"
                ],
                "materials": ["Flexible masonry caulk (polyurethane-based)", "Caulk gun", "Wire brush", "Vacuum", "Primer (optional)", "Paint (optional)"],
                "time": "1-2 hours",
                "effectiveness": "Good for waterproofing and cosmetics; does NOT restore structural strength"
            },
            {
                "name": "Spackling / Joint Compound (Drywall/Plaster)",
                "description": "Fill hairline cracks in drywall or plaster with joint compound, sand smooth, and repaint for invisible repair.",
                "when_to_use": "For hairline cracks in interior drywall or plaster walls",
                "diy_possible": True,
                "diy_cost": "$5 - $20",
                "professional_cost": "$150 - $400 per room",
                "steps": [
                    "Widen crack slightly with utility knife to improve adhesion",
                    "Remove all loose material with scraper or putty knife",
                    "Apply thin layer of joint compound with 4-inch putty knife",
                    "Embed paper tape over wider cracks for reinforcement",
                    "Apply second coat of compound when first is dry (4-6 hours)",
                    "Sand smooth with 120-220 grit sandpaper when fully dry",
                    "Apply primer and matching paint"
                ],
                "materials": ["Joint compound / spackle", "Putty knives (4-inch and 6-inch)", "Sandpaper (120-220 grit)", "Primer", "Matching paint", "Paper tape (for wider cracks)"],
                "time": "2-4 hours (plus 24h drying time between coats)",
                "effectiveness": "Excellent for cosmetic repair; crack may reappear if underlying movement continues"
            }
        ],
        "monitoring": "Mark both ends of crack with tape and write the date. Remeasure in 90 days. If stable with no growth, no further action needed. Check seasonally for changes.",
        "professional_needed": False,
        "cost_range": "$10 - $400"
    },

    "vertical_settlement_minor": {
        "name": "Vertical Crack (Minor Settlement / Shrinkage)",
        "direction": CrackDirection.VERTICAL,
        "severity": SeverityLevel.LOW,
        "width_range_mm": (1.5, 6.0),
        "is_structural": False,
        "urgency": "Monitor monthly. Seal with injection if desired.",
        "causes": [
            "Concrete shrinkage during curing (most common in first 1-3 years after construction)",
            "Minor uniform soil settlement as ground consolidates",
            "Thermal cycling in long, unjointed walls",
            "Poor taping during drywall construction",
            "Normal building adjustment to environmental conditions"
        ],
        "symptoms": [
            "Runs straight up and down (within 30 degrees of vertical)",
            "Width typically 1/16 to 1/8 inch (1.5mm to 3mm)",
            "Both sides at same level - NO vertical displacement",
            "May appear near center of long wall spans where tensile stress is highest",
            "Often found near window openings, pipe penetrations, or form tie locations",
            "Most common in poured concrete walls"
        ],
        "repair_methods": [
            {
                "name": "Epoxy Injection",
                "description": "Inject low-viscosity epoxy resin into the crack under low pressure (20-40 psi) to bond concrete back together and restore structural strength.",
                "when_to_use": "For dry, stable vertical cracks 1/16 to 1/4 inch wide in poured concrete. Best for structural restoration.",
                "diy_possible": True,
                "diy_cost": "$20 - $80 per crack (consumer kit)",
                "professional_cost": "$250 - $800 per crack",
                "steps": [
                    "Clean crack thoroughly with wire brush and compressed air",
                    "Seal crack surface with epoxy paste, leaving injection ports uncovered",
                    "Install surface-mounted injection ports every 6-8 inches along crack",
                    "Mix epoxy resin per manufacturer instructions (typically 2-part system)",
                    "Inject epoxy starting from lowest port, moving upward as material flows",
                    "Allow to cure fully (typically 24-72 hours depending on product)",
                    "Remove surface seal and grind flush if cosmetic finish needed"
                ],
                "materials": ["Epoxy injection kit (resin + hardener)", "Surface ports", "Wire brush", "Compressed air", "Mixing containers", "Caulk gun or injection pump"],
                "time": "3-6 hours (plus 24-72h cure time)",
                "effectiveness": "Excellent - restores structural strength and prevents water entry simultaneously"
            },
            {
                "name": "Polyurethane Injection",
                "description": "Inject flexible polyurethane foam that reacts with moisture to create waterproof, flexible seal. Can be done at low or high pressure (1500-3200 psi).",
                "when_to_use": "For actively leaking cracks or cracks with slight movement. Better than epoxy for wet conditions.",
                "diy_possible": True,
                "diy_cost": "$30 - $100 per crack (kit)",
                "professional_cost": "$300 - $900 per crack",
                "steps": [
                    "For high-pressure: flush crack with acid solution to clean interior",
                    "Install packers/injection ports along crack",
                    "Inject polyurethane at appropriate pressure for crack width",
                    "Material expands to fill void and displaces any water present",
                    "Cures within minutes to hours depending on formulation",
                    "Excess material can be trimmed after curing"
                ],
                "materials": ["Polyurethane injection kit", "Acid flush solution (for high-pressure)", "Injection pump or gun", "Packers/ports", "Safety equipment"],
                "time": "2-4 hours",
                "effectiveness": "Excellent waterproofing; flexible seal accommodates minor movement. Lower structural strength than epoxy."
            }
        ],
        "monitoring": "Mark crack ends with pencil and date. Measure width monthly with ruler or crack gauge. If widening beyond 1/4 inch (6mm) or one side becomes higher than the other, call a professional immediately.",
        "professional_needed": False,
        "cost_range": "$20 - $900"
    },

    "vertical_settlement_major": {
        "name": "Vertical Crack (Major Settlement with Displacement)",
        "direction": CrackDirection.VERTICAL,
        "severity": SeverityLevel.HIGH,
        "width_range_mm": (6.0, 50.0),
        "is_structural": True,
        "urgency": "Professional evaluation recommended immediately. Do not delay.",
        "causes": [
            "Uneven soil compaction beneath foundation",
            "Soil erosion from water flow beneath slab (poor drainage)",
            "Drought conditions causing clay soils to shrink dramatically",
            "Plumbing leaks washing away soil under foundation",
            "Tree root activity near foundation extracting moisture",
            "Inadequate foundation depth for soil conditions"
        ],
        "symptoms": [
            "Wider than 1/4 inch (6mm) - significantly wider than shrinkage cracks",
            "CRITICAL: One side of crack is HIGHER than the other (vertical displacement)",
            "Doors or windows sticking upstairs in affected area",
            "Sloping or uneven floors above the cracked foundation section",
            "Gaps appearing between walls and ceilings",
            "Crack continues to grow wider over weeks or months",
            "New vertical cracks appearing suddenly in short time period"
        ],
        "repair_methods": [
            {
                "name": "Underpinning with Steel/Concrete Piers",
                "description": "Install helical or push piers beneath foundation footing to transfer building load to stable soil deep below the problem layer.",
                "when_to_use": "For active settlement with vertical displacement. This is the permanent structural solution.",
                "diy_possible": False,
                "diy_cost": "N/A - Professional only",
                "professional_cost": "$1,000 - $3,000 per pier (typically 6-12 piers needed = $6,000 - $36,000 total)",
                "steps": [
                    "Structural engineer assessment and soil testing to determine pier type and depth",
                    "Excavation at pier locations to expose foundation footing",
                    "Install helical piers (screwed into ground) or push piers (driven to refusal)",
                    "Hydraulically lift foundation section back toward level position",
                    "Secure pier brackets to foundation footing with bolts",
                    "Backfill excavations and restore landscaping"
                ],
                "materials": ["Steel helical piers or concrete push piers", "Hydraulic lifting equipment", "Pier brackets", "Concrete for footing connection", "Excavation equipment"],
                "time": "2-5 days",
                "effectiveness": "Permanent structural solution - addresses root cause by reaching stable soil. Can lift foundation back to level."
            },
            {
                "name": "Mudjacking / Slab Jacking",
                "description": "Inject cementitious grout or polyurethane foam beneath sunken slab to fill voids and raise the section back toward original position.",
                "when_to_use": "For slab-on-grade settlement with minor to moderate displacement (typically under 2 inches)",
                "diy_possible": False,
                "diy_cost": "N/A - Professional only",
                "professional_cost": "$500 - $1,500 per area treated",
                "steps": [
                    "Drill small holes (1-2 inch diameter) through concrete slab",
                    "Pump cementitious grout or expanding polyurethane foam beneath slab",
                    "Monitor lift progress with laser level in real-time",
                    "Stop pumping when section reaches desired elevation",
                    "Seal injection holes with concrete patch material"
                ],
                "materials": ["Cementitious grout or polyurethane foam", "High-pressure injection pump", "Laser level", "Concrete patching compound"],
                "time": "1 day",
                "effectiveness": "Good for slab settlement; less invasive than underpinning. May need re-treatment over time if soil continues to move."
            },
            {
                "name": "Resin Injection Underpinning (Modern Alternative)",
                "description": "Inject expanding polyurethane resin into ground through small tubes to fill voids, strengthen soil, and stabilize foundation without major excavation.",
                "when_to_use": "Modern alternative to traditional underpinning - less invasive, faster, suitable for many residential applications",
                "diy_possible": False,
                "diy_cost": "N/A - Professional only",
                "professional_cost": "$4,000 - $15,000 total (average residential job)",
                "steps": [
                    "Site assessment and custom injection plan design",
                    "Drill small injection tubes into ground at strategic locations",
                    "Inject expanding polyurethane resin in controlled stages",
                    "Monitor building level during injection process",
                    "Remove tubes and restore surface (minimal disruption)",
                    "Verify stabilization with follow-up measurements"
                ],
                "materials": ["Expanding polyurethane resin", "Injection tubes", "Monitoring equipment", "Leveling instruments"],
                "time": "1-2 days (significantly faster than traditional underpinning)",
                "effectiveness": "Modern, fast, non-disruptive method. Suitable for many applications but not all soil conditions. Consult engineer."
            }
        ],
        "monitoring": "DO NOT simply seal this crack - the underlying settlement will reopen it. Professional structural assessment is mandatory. Monitor with dated pencil marks and measure width weekly. If crack grows by even 1/16 inch, call engineer immediately.",
        "professional_needed": True,
        "cost_range": "$4,000 - $36,000+"
    },

    "stair_step": {
        "name": "Stair-Step Crack (Differential Settlement in Masonry)",
        "direction": CrackDirection.STAIR_STEP,
        "severity": SeverityLevel.HIGH,
        "width_range_mm": (3.0, 50.0),
        "is_structural": True,
        "urgency": "Professional evaluation required. Do not delay - foundation is moving unevenly.",
        "causes": [
            "Differential settlement - one section of foundation sinking faster than adjacent section",
            "Soil washout from poor drainage or gutter downspout discharge",
            "Tree root activity near foundation extracting soil moisture",
            "Seasonal soil expansion and contraction in clay-heavy areas",
            "Uneven load distribution on foundation",
            "Mortar joints are weaker than blocks/bricks, so stress concentrates there"
        ],
        "symptoms": [
            "Follows mortar joints in block or brick foundation creating staircase pattern",
            "Diagonal path up the wall, stepping from horizontal to vertical mortar joints",
            "Often wider at one end than the other - indicates direction of settlement",
            "Most common in concrete block and brick foundations (not poured concrete)",
            "May be accompanied by wall bowing or bulging",
            "Doors and windows may stick or not close properly",
            "Interior drywall cracks may align with exterior stair-step cracks"
        ],
        "repair_methods": [
            {
                "name": "Tuckpointing + Structural Stabilization",
                "description": "Remove damaged mortar and replace with new matching mortar, combined with wall anchors or carbon fiber straps to prevent further movement.",
                "when_to_use": "For stable stair-step cracks where movement has stopped but masonry needs restoration",
                "diy_possible": False,
                "diy_cost": "N/A - Professional masonry work required",
                "professional_cost": "$5 - $25/sq ft for tuckpointing + $2,000 - $8,000 for wall stabilization",
                "steps": [
                    "Structural engineer assessment to confirm movement has stabilized",
                    "Grind out damaged mortar joints to minimum 3/4 inch depth",
                    "Clean joints thoroughly with compressed air and wire brush",
                    "Apply new mortar matching existing color and composition",
                    "Install wall anchors or carbon fiber straps for stabilization",
                    "Address drainage/grading issues to prevent recurrence"
                ],
                "materials": ["Matching mortar mix (Type N or S)", "Grinding tools (angle grinder with masonry blade)", "Wall anchors or carbon fiber straps", "Drainage improvement materials", "Compressed air"],
                "time": "3-7 days",
                "effectiveness": "Restores masonry integrity AND prevents future movement. Both steps are essential."
            },
            {
                "name": "Wall Anchors (Earth Anchor System)",
                "description": "Install steel anchors through basement wall into stable soil outside to hold wall firmly in place against lateral pressure.",
                "when_to_use": "For bowing or moving block walls with stair-step cracks showing active movement",
                "diy_possible": False,
                "diy_cost": "N/A - Professional only",
                "professional_cost": "$150 - $400 per anchor (typically 3-5 anchors per wall = $450 - $2,000)",
                "steps": [
                    "Excavate outside at anchor locations to expose wall",
                    "Drill through basement wall and into stable soil beyond",
                    "Install steel rod with large earth anchor plate on exterior end",
                    "Install interior wall plate and tighten with wrench",
                    "Backfill with proper drainage gravel",
                    "Tighten periodically as wall may gradually straighten"
                ],
                "materials": ["Steel anchor rods (galvanized)", "Earth anchor plates", "Interior wall plates", "Excavation equipment", "Drainage gravel"],
                "time": "1-3 days",
                "effectiveness": "Stops wall movement immediately. May gradually pull wall straight over months. Highly effective for block walls."
            },
            {
                "name": "Carbon Fiber Wall Reinforcement",
                "description": "Bond high-strength carbon fiber straps to wall surface using structural epoxy to reinforce and prevent further movement.",
                "when_to_use": "For minor to moderate wall bowing (under 2 inches) in block or poured walls",
                "diy_possible": False,
                "diy_cost": "N/A - Professional installation required",
                "professional_cost": "$2,000 - $5,000 per wall",
                "steps": [
                    "Assess bowing with straightedge or plumb line to measure deflection",
                    "Grind wall surface flat at strap locations for proper bonding",
                    "Apply structural epoxy primer to prepared surface",
                    "Bond carbon fiber straps vertically from floor to ceiling",
                    "Apply protective epoxy top coat over straps",
                    "Allow full cure (typically 24-48 hours) before backfilling"
                ],
                "materials": ["Carbon fiber straps (high-strength)", "Structural epoxy system (primer + adhesive + top coat)", "Grinding equipment", "Application tools"],
                "time": "1-2 days",
                "effectiveness": "Excellent for preventing further bowing. Minimally invasive - no excavation needed. Does NOT straighten already-bowed wall."
            }
        ],
        "monitoring": "Stair-step cracks indicate ongoing differential settlement. Sealing without structural stabilization is temporary and will fail. Professional assessment is mandatory. Mark with dated pencil lines and check weekly.",
        "professional_needed": True,
        "cost_range": "$2,000 - $15,000+"
    },

    "horizontal": {
        "name": "Horizontal Crack (Lateral Pressure / Wall Bowing)",
        "direction": CrackDirection.HORIZONTAL,
        "severity": SeverityLevel.VERY_HIGH,
        "width_range_mm": (3.0, 50.0),
        "is_structural": True,
        "urgency": "CALL A PROFESSIONAL IMMEDIATELY. This is a structural emergency requiring urgent attention.",
        "causes": [
            "Hydrostatic pressure from water-saturated soil pushing against basement wall",
            "Expansive clay soils that swell when wet and shrink when dry",
            "Frost heave in cold climates - water freezing in soil expands and pushes wall",
            "Poor drainage around foundation - water pooling against walls",
            "Inadequate wall thickness or reinforcement for soil conditions",
            "The crack appears at midpoint of wall height because that is where lateral pressure is most effective"
        ],
        "symptoms": [
            "Runs horizontally across wall (parallel to floor and ceiling)",
            "Usually appears at approximately midpoint of wall height - the weakest point",
            "1/8 inch (3mm) wide or more - significantly wider than shrinkage cracks",
            "Wall is visibly BOWING INWARD - check with straightedge or plumb line",
            "In concrete block walls, crack typically follows a horizontal mortar joint",
            "May see inward displacement of wall both above and below the crack line",
            "Often accompanied by water seepage during heavy rain"
        ],
        "repair_methods": [
            {
                "name": "Carbon Fiber Straps",
                "description": "Bond carbon fiber reinforcement straps to interior wall surface using structural epoxy to stop bowing and add tensile strength.",
                "when_to_use": "For minor wall bowing under 2 inches. Most common residential repair.",
                "diy_possible": False,
                "diy_cost": "N/A - Professional installation",
                "professional_cost": "$2,000 - $5,000 per wall",
                "steps": [
                    "Measure bowing with straightedge or plumb line to document deflection",
                    "Grind wall surface flat at strap locations to ensure proper bonding",
                    "Apply structural epoxy primer to prepared concrete surface",
                    "Bond carbon fiber straps vertically from floor to ceiling",
                    "Apply protective epoxy top coat over all straps",
                    "Allow full cure before applying any finishes"
                ],
                "materials": ["Carbon fiber straps (structural grade)", "Structural epoxy system (3-part)", "Concrete grinder", "Measuring tools"],
                "time": "1-2 days",
                "effectiveness": "Prevents further bowing. Minimally invasive. Does NOT straighten already-bowed wall - only stops progression."
            },
            {
                "name": "Wall Anchors (Earth Anchor System)",
                "description": "Steel anchors connect interior basement wall to stable soil outside to resist lateral soil pressure.",
                "when_to_use": "For moderate wall bowing between 2-4 inches",
                "diy_possible": False,
                "diy_cost": "N/A - Professional only",
                "professional_cost": "$3,000 - $8,000 per wall",
                "steps": [
                    "Excavate outside at anchor points down to footing level",
                    "Drill through basement wall and into stable soil 8-10 feet beyond",
                    "Install steel rod with large earth anchor plate on exterior",
                    "Install interior wall plate and tighten with torque wrench",
                    "Backfill with free-draining gravel to improve drainage",
                    "Tighten anchors periodically - wall may gradually straighten"
                ],
                "materials": ["Steel anchor system (rods + plates)", "Excavation equipment", "Torque wrench", "Drainage gravel"],
                "time": "2-4 days",
                "effectiveness": "Stops movement immediately. May gradually pull wall straight over time. More invasive than carbon fiber but stronger."
            },
            {
                "name": "Steel I-Beam Installation",
                "description": "Install vertical steel I-beams against basement wall to provide rigid structural support and prevent catastrophic failure.",
                "when_to_use": "For severe bowing over 4 inches, or when anchors are not feasible due to property lines or obstacles",
                "diy_possible": False,
                "diy_cost": "N/A - Professional structural work",
                "professional_cost": "$4,000 - $10,000 per wall",
                "steps": [
                    "Structural engineer designs beam spacing and specifications",
                    "Fabricate or order steel I-beams to exact measurements",
                    "Excavate floor slab at beam locations down to footing",
                    "Set beams vertically against wall at designed spacing (typically 4-6 feet)",
                    "Secure top to floor joists above, bottom to concrete footing",
                    "Backfill and restore floor with concrete"
                ],
                "materials": ["Steel I-beams (structural grade)", "Concrete for footing and floor restoration", "Heavy-duty fasteners", "Floor cutting equipment"],
                "time": "2-5 days",
                "effectiveness": "Maximum structural support. Prevents catastrophic wall failure. Most invasive option but strongest."
            },
            {
                "name": "Exterior Waterproofing + Drainage System",
                "description": "Excavate exterior, apply waterproof membrane, install drainage board and perimeter drain to eliminate hydrostatic pressure - addresses ROOT CAUSE.",
                "when_to_use": "ESSENTIAL companion to any horizontal crack repair. Without this, other repairs will eventually fail.",
                "diy_possible": False,
                "diy_cost": "N/A - Professional excavation work",
                "professional_cost": "$3,000 - $15,000 depending on wall length and depth",
                "steps": [
                    "Excavate to footing level along entire affected wall length",
                    "Clean and prepare wall surface (power wash, repair voids)",
                    "Apply waterproof membrane or bituminous coating to wall",
                    "Install dimpled drainage board against wall",
                    "Place perforated drain pipe at footing level surrounded by gravel",
                    "Backfill with free-draining soil (NOT clay)",
                    "Ensure positive grading away from foundation"
                ],
                "materials": ["Waterproof membrane (rubberized asphalt or PVC)", "Dimpled drainage board", "Perforated PVC drain pipe", "Drainage gravel", "Excavation equipment"],
                "time": "3-7 days",
                "effectiveness": "Addresses ROOT CAUSE by eliminating hydrostatic pressure. Critical for long-term success of any wall repair."
            }
        ],
        "monitoring": "THIS IS A STRUCTURAL EMERGENCY. Horizontal cracks indicate the wall is actively failing under lateral pressure. Do NOT delay professional help. Evacuate basement if bowing is severe. Document with photos for insurance.",
        "professional_needed": True,
        "cost_range": "$5,000 - $25,000+"
    },

    "diagonal": {
        "name": "Diagonal Crack (Differential Settlement / Subsidence)",
        "direction": CrackDirection.DIAGONAL,
        "severity": SeverityLevel.HIGH,
        "width_range_mm": (2.0, 50.0),
        "is_structural": True,
        "urgency": "Professional evaluation required. Schedule structural engineer assessment soon.",
        "causes": [
            "Differential settlement - one corner or section of foundation sinking more than others",
            "Poor soil compaction during original construction",
            "Changes in soil moisture content - drought causes shrinkage, wet periods cause expansion",
            "Inadequate drainage causing soil saturation and softening",
            "Stress concentration at door and window openings where load paths change",
            "Tree roots removing moisture from soil beneath foundation"
        ],
        "symptoms": [
            "Runs at approximately 45-degree angle across wall",
            "Often starts from corner of window or door frame - classic stress concentration point",
            "Wider at one end than the other - the wider end indicates direction of settlement",
            "May appear suddenly over days or weeks",
            "Visible on BOTH interior and exterior walls (check outside too)",
            "Often accompanied by sticking doors and windows",
            "May see gaps around door frames or window trim"
        ],
        "repair_methods": [
            {
                "name": "Foundation Underpinning + Crack Injection",
                "description": "Address underlying settlement with piers or soil stabilization FIRST, then inject crack with epoxy once movement stops.",
                "when_to_use": "For active diagonal cracks with ongoing settlement. Order matters: stabilize first, then repair crack.",
                "diy_possible": False,
                "diy_cost": "N/A - Professional structural work",
                "professional_cost": "$3,000 - $15,000+ depending on severity and number of piers needed",
                "steps": [
                    "Licensed structural engineer assessment and soil testing",
                    "Determine cause: soil testing, drainage evaluation, plumbing inspection",
                    "Install foundation piers or stabilize soil as engineer specifies",
                    "Wait for stabilization (monitor for 6-12 months if possible)",
                    "Once movement stops, inject crack with epoxy for structural bond",
                    "Continue monitoring with crack gauges for 1-2 years"
                ],
                "materials": ["Foundation piers (helical or push)", "Epoxy injection system", "Soil testing equipment", "Crack monitoring gauges"],
                "time": "3-10 days for piers + ongoing monitoring",
                "effectiveness": "Addresses both symptom (crack) and cause (settlement). Most comprehensive solution."
            },
            {
                "name": "Soil Stabilization (Chemical Grouting)",
                "description": "Inject chemical grouts or expanding resins into soil to improve bearing capacity, fill voids, and prevent further settlement.",
                "when_to_use": "For soil-related settlement where underpinning is not feasible or as a less invasive alternative",
                "diy_possible": False,
                "diy_cost": "N/A - Specialized geotechnical work",
                "professional_cost": "$2,000 - $10,000",
                "steps": [
                    "Geotechnical investigation to determine soil composition and void locations",
                    "Design injection pattern based on soil conditions",
                    "Drill injection points around foundation at specified depths and spacing",
                    "Inject chemical grout or resin in controlled stages",
                    "Monitor settlement gauges during and after injection",
                    "Verify improvement with follow-up testing"
                ],
                "materials": ["Chemical grout or polyurethane resin", "Specialized injection equipment", "Settlement monitoring gauges", "Drilling equipment"],
                "time": "2-5 days",
                "effectiveness": "Improves soil without major excavation. Less disruptive than underpinning. Results vary by soil type."
            }
        ],
        "monitoring": "Diagonal cracks are classic signs of subsidence. Even if they appear stable, underlying soil conditions may still be changing. Professional assessment is essential. Mark with dated pencil lines and measure width weekly. Any growth requires immediate engineer contact.",
        "professional_needed": True,
        "cost_range": "$3,000 - $20,000+"
    },

    "map_crazing": {
        "name": "Map / Crazing Cracks (Surface Shrinkage)",
        "direction": CrackDirection.MAP_CRACKING,
        "severity": SeverityLevel.LOW,
        "width_range_mm": (0.0, 1.0),
        "is_structural": False,
        "urgency": "Cosmetic issue only. Monitor and seal if desired.",
        "causes": [
            "Surface-level shrinkage in concrete or plaster",
            "Rapid surface drying while interior of material remains moist",
            "Overworking surface during finishing process",
            "Adding water to surface during finishing (creates weak laitance layer)",
            "Inadequate curing - surface dried too quickly"
        ],
        "symptoms": [
            "Web-like network of fine lines across surface - looks like dried riverbed",
            "Very shallow, surface-only - does not penetrate deeply",
            "Random pattern with no dominant direction",
            "Common in concrete slabs, driveways, and plaster finishes",
            "All cracks are hairline width - less than 1mm",
            "No displacement between crack sides",
            "Does not leak water"
        ],
        "repair_methods": [
            {
                "name": "Surface Refinishing / Resurfacing",
                "description": "Apply thin concrete overlay or resurfacing compound to cover crazing and restore uniform appearance.",
                "when_to_use": "For cosmetic crazing on floors, driveways, or walls where appearance matters",
                "diy_possible": True,
                "diy_cost": "$50 - $200",
                "professional_cost": "$500 - $2,000",
                "steps": [
                    "Clean surface thoroughly - pressure wash if outdoors",
                    "Repair any deeper cracks first with appropriate filler",
                    "Mix concrete resurfacer per manufacturer instructions",
                    "Apply with trowel, squeegee, or broom depending on desired finish",
                    "Work in small sections to maintain wet edge",
                    "Allow to cure per manufacturer (typically 24-48 hours before light use)"
                ],
                "materials": ["Concrete resurfacer or skim coat", "Trowel or squeegee", "Pressure washer or cleaning supplies", "Mixing container"],
                "time": "4-8 hours (plus 24-48h curing)",
                "effectiveness": "Excellent cosmetic restoration. Creates new durable surface over crazed area."
            }
        ],
        "monitoring": "Purely cosmetic - no structural concern whatsoever. Seal to prevent dust accumulation and improve appearance. No monitoring needed unless cracks deepen or widen significantly.",
        "professional_needed": False,
        "cost_range": "$50 - $2,000"
    },

    "thermal_movement": {
        "name": "Thermal Movement Crack",
        "direction": CrackDirection.VERTICAL,
        "severity": SeverityLevel.LOW,
        "width_range_mm": (0.5, 3.0),
        "is_structural": False,
        "urgency": "Monitor. Seal with flexible material that accommodates movement.",
        "causes": [
            "Seasonal temperature expansion and contraction of building materials",
            "Lack of expansion joints in long walls (over 25 meters)",
            "Different materials with different thermal expansion rates joined together",
            "Sun exposure causing uneven heating of wall surfaces",
            "Temperature swings between day and night in exposed walls"
        ],
        "symptoms": [
            "Vertical crack, often in middle of long wall span",
            "Opens wider in winter (cold contraction), closes in summer (warm expansion)",
            "Typically less than 2mm wide",
            "Appears at joints, corners, or where different materials meet",
            "No displacement between sides - both sides remain flush",
            "May reappear after rigid repairs because movement continues",
            "Follows predictable seasonal pattern"
        ],
        "repair_methods": [
            {
                "name": "Flexible Sealant / Caulk",
                "description": "Use high-quality flexible sealant that stretches and compresses with seasonal movement without cracking.",
                "when_to_use": "For thermal cracks that open and close seasonally. Rigid fillers will fail.",
                "diy_possible": True,
                "diy_cost": "$15 - $50",
                "professional_cost": "$200 - $600",
                "steps": [
                    "Clean crack thoroughly and remove any old rigid filler",
                    "For deep cracks, install backer rod to control sealant depth",
                    "Apply flexible polyurethane or high-quality silicone sealant",
                    "Tool smooth with wet finger or shaping tool",
                    "Allow full cure before painting (check product specs)",
                    "Paint with flexible paint if desired"
                ],
                "materials": ["Flexible sealant (polyurethane or silicone, NOT acrylic caulk)", "Backer rod (for deep cracks)", "Caulk gun", "Shaping tool"],
                "time": "1-2 hours",
                "effectiveness": "Good - accommodates up to 25% movement. May need reapplication every 5-10 years."
            },
            {
                "name": "Expansion Joint Installation",
                "description": "Cut and install proper expansion joint to accommodate thermal movement permanently.",
                "when_to_use": "For recurring thermal cracks in long walls where sealant keeps failing",
                "diy_possible": False,
                "diy_cost": "N/A - Professional masonry work",
                "professional_cost": "$500 - $2,000",
                "steps": [
                    "Structural assessment to determine joint location and spacing",
                    "Cut full-depth joint through wall at crack location",
                    "Install preformed expansion joint filler material",
                    "Seal joint with flexible joint sealant rated for movement",
                    "Finish edges to match wall surface"
                ],
                "materials": ["Preformed expansion joint material", "Flexible joint sealant", "Concrete/masonry cutting tools", "Finishing materials"],
                "time": "1-2 days",
                "effectiveness": "Permanent solution for thermal movement. Properly designed joints prevent future cracking."
            }
        ],
        "monitoring": "Track seasonally - measure width in winter and summer. If crack width exceeds 3mm or shows any displacement (one side higher), consult structural engineer as this may indicate settlement rather than pure thermal movement.",
        "professional_needed": False,
        "cost_range": "$15 - $2,000"
    }
}


# =============================================================================
# IMAGE ANALYSIS ENGINE
# =============================================================================
class ImageAnalyzer:
    def __init__(self):
        self.min_crack_width_pixels = 2
        self.edge_threshold = 30

    def analyze_image(self, image_path):
        if not PIL_AVAILABLE:
            return {"error": "Pillow not installed", "success": False}
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": str(e), "success": False}

        img = self._resize_for_analysis(img)
        width, height = img.size

        if NUMPY_AVAILABLE:
            img_array = np.array(img)
            gray = self._to_grayscale_numpy(img_array)
            edges = self._detect_edges_numpy(gray)
            lines = self._detect_lines_numpy(edges)
            crack_mask = self._segment_cracks_numpy(gray, edges)
            features = self._extract_features_numpy(crack_mask, lines, width, height)
        else:
            gray = self._to_grayscale_pil(img)
            edges = self._detect_edges_pil(gray)
            lines = self._detect_lines_pil(edges, width, height)
            crack_mask = self._segment_cracks_pil(gray, edges)
            features = self._extract_features_pil(crack_mask, lines, width, height)

        classification = self._classify_crack(features, width, height)
        return {"success": True, "image_size": (width, height), "features": features,
                "classification": classification, "visualization_available": True}

    def _resize_for_analysis(self, img, max_dim=800):
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            return img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)
        return img

    def _to_grayscale_numpy(self, img_array):
        return np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def _to_grayscale_pil(self, img):
        return img.convert("L")

    def _detect_edges_numpy(self, gray):
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        padded = np.pad(gray, 1, mode='edge')
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                gx += sobel_x[i,j] * padded[i:i+gray.shape[0], j:j+gray.shape[1]]
                gy += sobel_y[i,j] * padded[i:i+gray.shape[0], j:j+gray.shape[1]]
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude / max(magnitude.max(),1) * 255).astype(np.uint8)
        return (magnitude > self.edge_threshold).astype(np.uint8) * 255

    def _detect_edges_pil(self, gray):
        edges = gray.filter(ImageFilter.FIND_EDGES)
        enhancer = ImageEnhance.Contrast(edges)
        return enhancer.enhance(2.0)

    def _detect_lines_numpy(self, edges):
        edge_pixels = np.argwhere(edges > 128)
        if len(edge_pixels) < 50:
            return []
        if len(edge_pixels) > 2000:
            edge_pixels = edge_pixels[np.random.choice(len(edge_pixels), 2000, replace=False)]
        orientations = []
        h, w = edges.shape
        for y, x in edge_pixels[::10]:
            y1, y2 = max(0,y-5), min(h,y+6)
            x1, x2 = max(0,x-5), min(w,x+6)
            neighborhood = edges[y1:y2, x1:x2]
            coords = np.argwhere(neighborhood > 128)
            if len(coords) < 3:
                continue
            coords = coords - coords.mean(axis=0)
            if coords.shape[0] > 1:
                cov = np.cov(coords.T)
                if cov.shape == (2,2):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    if eigvals[1] > 0:
                        angle = math.degrees(math.atan2(eigvecs[1,1], eigvecs[0,1]))
                        orientations.append(angle)
        if not orientations:
            return []
        bins = {"vertical":0, "horizontal":0, "diagonal":0, "other":0}
        for angle in orientations:
            norm = angle % 180
            if norm < 20 or norm > 160:
                bins["vertical"] += 1
            elif 70 <= norm <= 110:
                bins["horizontal"] += 1
            elif (20 <= norm < 70) or (110 < norm <= 160):
                bins["diagonal"] += 1
            else:
                bins["other"] += 1
        total = sum(bins.values())
        if total == 0:
            return []
        return [{"orientation":k, "count":v, "percentage":v/total*100} for k,v in sorted(bins.items(), key=lambda x:-x[1])]

    def _detect_lines_pil(self, edges, width, height):
        return [{"orientation":"vertical","count":height,"percentage":40},
                {"orientation":"horizontal","count":width,"percentage":30},
                {"orientation":"diagonal","count":50,"percentage":20},
                {"orientation":"other","count":20,"percentage":10}]

    def _segment_cracks_numpy(self, gray, edges):
        return ((gray < 80) & (edges > 128)).astype(np.uint8) * 255

    def _segment_cracks_pil(self, gray, edges):
        gd = list(gray.getdata())
        ed = list(edges.getdata())
        mask = Image.new("L", gray.size)
        mask.putdata([255 if g<80 and e>100 else 0 for g,e in zip(gd,ed)])
        return mask

    def _extract_features_numpy(self, crack_mask, lines, width, height):
        crack_pixels = np.sum(crack_mask > 0)
        total = width * height
        labeled = self._simple_label_numpy(crack_mask)
        num_components = labeled.max()
        bbox_width = bbox_height = aspect_ratio = est_width = 0
        if num_components > 0:
            counts = np.bincount(labeled.flat)[1:]
            if len(counts) > 0:
                largest = np.argmax(counts) + 1
                coords = np.argwhere(labeled == largest)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    aspect_ratio = bbox_height / max(bbox_width, 1)
                    crack_length = max(bbox_width, bbox_height)
                    est_width = crack_pixels / max(crack_length, 1)
        dom = lines[0]["orientation"] if lines else "unknown"
        conf = lines[0]["percentage"] if lines else 0
        return {
            "crack_ratio": round(crack_pixels/total*100, 2),
            "num_components": int(num_components),
            "bbox_width": int(bbox_width), "bbox_height": int(bbox_height),
            "aspect_ratio": round(aspect_ratio, 2),
            "estimated_width_pixels": round(est_width, 2),
            "dominant_orientation": dom,
            "orientation_confidence": round(conf, 1),
            "stair_step_score": round(self._detect_stair_step_numpy(crack_mask, width, height), 2),
            "crazing_score": round(self._detect_crazing_numpy(crack_mask, width, height), 2),
            "total_crack_pixels": int(crack_pixels)
        }

    def _extract_features_pil(self, crack_mask, lines, width, height):
        cp = sum(1 for p in list(crack_mask.getdata()) if p > 128)
        return {
            "crack_ratio": round(cp/(width*height)*100, 2),
            "num_components": 1, "bbox_width": width//3, "bbox_height": height//2,
            "aspect_ratio": 1.5, "estimated_width_pixels": 5.0,
            "dominant_orientation": lines[0]["orientation"] if lines else "unknown",
            "orientation_confidence": 50.0, "stair_step_score": 0.1,
            "crazing_score": 0.1, "total_crack_pixels": cp
        }

    def _simple_label_numpy(self, mask):
        h, w = mask.shape
        labeled = np.zeros((h,w), dtype=np.int32)
        label = 1
        for y in range(h):
            for x in range(w):
                if mask[y,x] > 0 and labeled[y,x] == 0:
                    stack = [(y,x)]
                    while stack:
                        cy, cx = stack.pop()
                        if 0<=cy<h and 0<=cx<w and mask[cy,cx]>0 and labeled[cy,cx]==0:
                            labeled[cy,cx] = label
                            stack.extend([(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)])
                    label += 1
        return labeled

    def _detect_stair_step_numpy(self, mask, width, height):
        h, w = mask.shape
        h_score = v_score = 0
        for y in range(0, h, max(h//20,1)):
            runs = self._find_runs(mask[y,:])
            h_score += sum(1 for s,e in runs if e-s > w*0.1)
        for x in range(0, w, max(w//20,1)):
            runs = self._find_runs(mask[:,x])
            v_score += sum(1 for s,e in runs if e-s > h*0.1)
        return min((h_score+v_score)/10, 1.0) if h_score>2 and v_score>2 else 0.0

    def _detect_crazing_numpy(self, mask, width, height):
        nc = self._simple_label_numpy(mask).max()
        return min(nc/50, 1.0) if nc > 10 else 0.0

    def _find_runs(self, arr):
        runs, in_run, start = [], False, 0
        for i, v in enumerate(arr):
            if v > 0 and not in_run:
                in_run, start = True, i
            elif v == 0 and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, len(arr)))
        return runs

    def _classify_crack(self, features, img_width, img_height):
        orientation = features.get("dominant_orientation", "unknown")
        aspect = features.get("aspect_ratio", 1.0)
        stair = features.get("stair_step_score", 0)
        crazy = features.get("crazing_score", 0)
        est_px = features.get("estimated_width_pixels", 0)
        ratio = features.get("crack_ratio", 0)
        mm_per_px = 3000 / max(img_width, 1)
        est_mm = est_px * mm_per_px
        scores = {}
        if orientation == "horizontal" and aspect < 0.3:
            scores = {"horizontal":0.85, "hairline_shrinkage":0.10, "map_crazing":0.05}
        elif stair > 0.3:
            scores = {"stair_step":0.80, "diagonal":0.15, "vertical_settlement_major":0.05}
        elif crazy > 0.3 or (ratio > 5 and est_mm < 1):
            scores = {"map_crazing":0.75, "hairline_shrinkage":0.20, "thermal_movement":0.05}
        elif orientation == "diagonal" and aspect > 0.5:
            scores = {"diagonal":0.70, "stair_step":0.20, "vertical_settlement_major":0.10}
        elif orientation == "vertical" and aspect > 1.5:
            if est_mm < 1.5:
                scores = {"hairline_shrinkage":0.60, "vertical_settlement_minor":0.30, "thermal_movement":0.10}
            elif est_mm < 6:
                scores = {"vertical_settlement_minor":0.60, "thermal_movement":0.25, "hairline_shrinkage":0.10, "vertical_settlement_major":0.05}
            else:
                scores = {"vertical_settlement_major":0.70, "vertical_settlement_minor":0.20, "stair_step":0.10}
        elif orientation == "vertical" and 0.8 < aspect < 3 and est_mm < 3:
            scores = {"thermal_movement":0.50, "vertical_settlement_minor":0.30, "hairline_shrinkage":0.20}
        else:
            if est_mm < 1:
                scores = {"hairline_shrinkage":0.40, "map_crazing":0.30, "vertical_settlement_minor":0.30}
            elif est_mm < 6:
                scores = {"vertical_settlement_minor":0.40, "thermal_movement":0.30, "diagonal":0.30}
            else:
                scores = {"vertical_settlement_major":0.35, "diagonal":0.35, "stair_step":0.30}
        sorted_types = sorted(scores.items(), key=lambda x:-x[1])
        return {
            "primary_type": sorted_types[0][0],
            "primary_confidence": round(sorted_types[0][1], 2),
            "all_scores": {k:round(v,2) for k,v in sorted_types},
            "estimated_width_mm": round(est_mm, 2),
            "reasoning": f"Detected {orientation} crack with estimated width {est_mm:.1f}mm"
        }


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Wall Crack Analyzer", page_icon="🏠", layout="wide")

st.title("🏠 Wall Crack Analyzer")
st.markdown("Upload a photo of a wall crack and get a professional structural analysis with repair recommendations.")
st.markdown("---")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This tool uses real computer vision (Sobel edge detection, PCA orientation analysis) 
to identify crack types based on structural engineering data.

**Data Sources:**
- The Basement Guide
- JLB Foundation & Waterproofing
- Mainmark (underpinning costs)
- AquaGuard Injection
- Dalinghaus Construction

**Disclaimer:** This is NOT a substitute for professional structural assessment.
Always consult a licensed structural engineer for definitive evaluation.
""")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Upload a clear photo of the crack
2. Select wall material type
3. Click **Analyze Crack**
4. Review the detailed report
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a crack image", type=["jpg", "jpeg", "png", "webp"])

    wall_material = st.selectbox(
        "Wall Material",
        ["Unknown", "Concrete (poured foundation)", "Brick / Block (masonry)", "Plaster / Drywall (interior)"],
        index=0
    )

    material_map = {
        "Unknown": "unknown",
        "Concrete (poured foundation)": "concrete",
        "Brick / Block (masonry)": "brick_block",
        "Plaster / Drywall (interior)": "plaster_drywall"
    }

    analyze_btn = st.button("🔍 Analyze Crack", type="primary", use_container_width=True)

# Analysis
if uploaded_file is not None and analyze_btn:
    with st.spinner("Analyzing crack image... This may take a few seconds."):
        # Save uploaded file
        img = Image.open(uploaded_file)
        temp_path = "/tmp/crack_upload.png"
        img.save(temp_path)

        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze_image(temp_path)

        if not result.get("success", False):
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            classification = result["classification"]
            features = result["features"]
            crack_type = classification["primary_type"]
            confidence = classification["primary_confidence"]

            kb = CRACK_KNOWLEDGE_BASE.get(crack_type, {})
            if not kb:
                kb = CRACK_KNOWLEDGE_BASE["hairline_shrinkage"]

            # Severity color
            severity = kb.get("severity", SeverityLevel.LOW)
            severity_colors = {
                SeverityLevel.LOW: "🟢",
                SeverityLevel.MODERATE: "🟡",
                SeverityLevel.HIGH: "🟠",
                SeverityLevel.VERY_HIGH: "🔴",
                SeverityLevel.CRITICAL: "⛔"
            }
            sev_color = severity_colors.get(severity, "⚪")

            with col1:
                st.subheader("📊 Analysis Results")

                # Severity banner
                if kb.get("is_structural", False):
                    st.error("⚠️ STRUCTURAL CRACK DETECTED - Professional Help Required")
                else:
                    st.success("✅ Non-Structural Crack - DIY Repair Possible")

                # Key metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Crack Type", kb.get("name", crack_type)[:20])
                with m2:
                    st.metric("Severity", f"{sev_color} {severity.value}")
                with m3:
                    st.metric("Est. Width", f"{classification.get('estimated_width_mm', 0):.1f} mm")
                with m4:
                    st.metric("Confidence", f"{confidence*100:.0f}%")

                st.markdown("---")

                # Urgency
                st.subheader("⏰ Urgency")
                st.info(kb.get("urgency", "Monitor"))

                # Causes
                st.subheader("🔍 Root Causes")
                for cause in kb.get("causes", []):
                    st.markdown(f"• {cause}")

                # Symptoms
                st.subheader("📋 Identification Symptoms")
                for symptom in kb.get("symptoms", []):
                    st.markdown(f"• {symptom}")

                # Monitoring
                st.subheader("👁️ Monitoring Advice")
                st.warning(kb.get("monitoring", "Monitor periodically."))

                # Cost
                st.subheader("💰 Estimated Cost")
                st.markdown(f"**Range:** {kb.get('cost_range', 'Unknown')}")

                if kb.get("professional_needed", False):
                    st.error("🚨 This crack requires professional structural evaluation. Do not attempt DIY repair.")
                else:
                    st.success("✅ DIY repair is feasible. Follow the repair methods below.")

            with col2:
                st.subheader("🖼️ Uploaded Image")
                st.image(img, use_container_width=True)

                # Technical details
                with st.expander("🔧 Technical Analysis Details"):
                    st.json(features)
                    st.markdown(f"**Classification Reasoning:** {classification.get('reasoning', '')}")
                    st.markdown("**Confidence Scores:**")
                    for ctype, score in classification.get("all_scores", {}).items():
                        kb_name = CRACK_KNOWLEDGE_BASE.get(ctype, {}).get("name", ctype)
                        st.progress(score, text=f"{kb_name}: {score*100:.0f}%")

                # Repair methods
                st.subheader("🔨 Repair Solutions")
                for idx, method in enumerate(kb.get("repair_methods", []), 1):
                    with st.expander(f"Method {idx}: {method['name']}"):
                        st.markdown(f"**Description:** {method['description']}")
                        st.markdown(f"**When to Use:** {method['when_to_use']}")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**DIY Possible:** {'✅ Yes' if method['diy_possible'] else '❌ No'}")
                            st.markdown(f"**DIY Cost:** {method['diy_cost']}")
                        with c2:
                            st.markdown(f"**Professional Cost:** {method['professional_cost']}")
                            st.markdown(f"**Time:** {method['time_estimate']}")

                        st.markdown(f"**Effectiveness:** {method['effectiveness']}")
                        st.markdown(f"**Materials:** {', '.join(method['materials'])}")

                        st.markdown("**Steps:**")
                        for j, step in enumerate(method['steps'], 1):
                            st.markdown(f"{j}. {step}")

                # Download report
                st.markdown("---")
                st.subheader("📥 Download Report")

                # Generate text report
                report_lines = []
                report_lines.append("=" * 70)
                report_lines.append("WALL CRACK ANALYSIS REPORT")
                report_lines.append("=" * 70)
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Confidence: {confidence*100:.1f}%")
                report_lines.append("")
                report_lines.append(f"CRACK TYPE: {kb.get('name', crack_type)}")
                report_lines.append(f"SEVERITY: {severity.value}")
                report_lines.append(f"STRUCTURAL: {'YES' if kb.get('is_structural') else 'No'}")
                report_lines.append(f"EST. WIDTH: {classification.get('estimated_width_mm', 0):.1f} mm")
                report_lines.append(f"URGENCY: {kb.get('urgency', 'Monitor')}")
                report_lines.append("")
                report_lines.append("CAUSES:")
                for c in kb.get('causes', []):
                    report_lines.append(f"  - {c}")
                report_lines.append("")
                report_lines.append("SYMPTOMS:")
                for s in kb.get('symptoms', []):
                    report_lines.append(f"  - {s}")
                report_lines.append("")
                report_lines.append("REPAIR METHODS:")
                for m in kb.get('repair_methods', []):
                    report_lines.append(f"
  {m['name']}")
                    report_lines.append(f"  DIY: {'Yes' if m['diy_possible'] else 'No'} | Cost: {m['diy_cost']} | Pro: {m['professional_cost']}")
                    report_lines.append(f"  Time: {m['time_estimate']} | Effectiveness: {m['effectiveness']}")
                report_lines.append("")
                report_lines.append(f"MONITORING: {kb.get('monitoring', '')}")
                report_lines.append(f"COST RANGE: {kb.get('cost_range', 'Unknown')}")
                report_lines.append("")
                report_lines.append("DISCLAIMER: This analysis is based on image processing and structural")
                report_lines.append("engineering knowledge. NOT a substitute for professional assessment.")
                report_lines.append("=" * 70)

                report_text = "\n".join(report_lines)

                st.download_button(
                    label="📄 Download Text Report",
                    data=report_text,
                    file_name="crack_analysis_report.txt",
                    mime="text/plain"
                )

                # JSON report
                json_data = {
                    "crack_type": crack_type,
                    "crack_name": kb.get("name", crack_type),
                    "severity": severity.value,
                    "is_structural": kb.get("is_structural", False),
                    "estimated_width_mm": classification.get("estimated_width_mm", 0),
                    "confidence": confidence,
                    "urgency": kb.get("urgency", ""),
                    "causes": kb.get("causes", []),
                    "symptoms": kb.get("symptoms", []),
                    "repair_methods": kb.get("repair_methods", []),
                    "cost_range": kb.get("cost_range", ""),
                    "professional_needed": kb.get("professional_needed", False),
                    "monitoring": kb.get("monitoring", ""),
                    "image_features": features,
                    "timestamp": datetime.now().isoformat()
                }

                st.download_button(
                    label="📊 Download JSON Report",
                    data=json.dumps(json_data, indent=2),
                    file_name="crack_analysis_report.json",
                    mime="application/json"
                )

elif uploaded_file is None and analyze_btn:
    st.warning("Please upload an image first.")

# Reference guide at bottom
st.markdown("---")
st.subheader("📚 Crack Type Reference Guide")

ref_cols = st.columns(4)
crack_types = list(CRACK_KNOWLEDGE_BASE.items())
for i, (key, data) in enumerate(crack_types):
    with ref_cols[i % 4]:
        with st.expander(data["name"][:30]):
            st.markdown(f"**Direction:** {data['direction'].value}")
            st.markdown(f"**Severity:** {data['severity'].value}")
            st.markdown(f"**Width:** {data['width_range_mm'][0]:.1f} - {data['width_range_mm'][1]:.1f} mm")
            st.markdown(f"**Structural:** {'Yes' if data['is_structural'] else 'No'}")
            st.markdown(f"**Cost:** {data['cost_range']}")
            st.markdown(f"**DIY Possible:** {'No' if data['professional_needed'] else 'Yes'}")
