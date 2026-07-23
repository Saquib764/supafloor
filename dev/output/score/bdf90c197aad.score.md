# Score report — 3-room

Source: `../plans/bdf90c197aad.json`

![3-room](bdf90c197aad.png)

This floor plan scores **2.72/10 (Poor)** for “3-room” with 10 rooms, 1176.0 ft² total room area. It fails 4 hard rule(s): Reachability, Window Lines, Bathroom Accessible, Kitchen Accessible. Strong areas include living size, circulation efficiency, room proportions. Weaker areas include natural light, wet core compactness, storage adequacy. Highest category is geometry (10.0); lowest is daylight (2.7).

## 3-room

**Overall score: 2.72 / 10 — Poor**

**Plan:** 3-room  
**Rooms (10):** Living (180.0 ft²), Room 1 (174.0 ft²), Room 2 (184.0 ft²), Room 1 Bath (96.0 ft²), Room 1 Toilet (90.0 ft²), Room 2 Bath (68.0 ft²), Kitchen (94.0 ft²), Dining (140.0 ft²), Hallway (60.0 ft²), Balcony (90.0 ft²)  
**Doors:** 4 line openings · **Windows:** 2 line openings


### Hard failures (plan rejected by geometry/topology rules)
- **Reachability** (reachability): Unreachable: Room 1, Room 2, Room 1 Bath, Room 2 Bath, Kitchen, Dining — *Repair:* Add doors connecting unreachable rooms to circulation
- **Window Lines** (windows_on_exterior): Missing exterior glazing: ['Room 1', 'Room 2', 'Dining'] — *Repair:* Add window line segments on exterior edges of living/bedrooms
- **Bathroom Accessible** (bathroom_accessible): Bathroom lacks door connections — *Repair:* Connect bathroom to hall or bedroom
- **Kitchen Accessible** (kitchen_accessible): Kitchen not connected — *Repair:* Connect kitchen to living/dining

### Category scores

| Category | Score | Rating | Weight |
|---|---:|---|---:|
| Geometry | 10.00 | Excellent | 0.15 |
| Topology | 3.67 | Poor | 0.10 |
| Construction | 4.32 | Weak | 0.10 |
| Privacy | 7.50 | Good | 0.11 |
| Circulation | 8.31 | Good | 0.10 |
| Daylight | 2.66 | Poor | 0.13 |
| Architecture | 7.74 | Good | 0.13 |
| Aesthetics | 7.85 | Good | 0.18 |

### Rule details

- **Polygon Validity** [HARD/PASS] — **10.0/10** (Excellent): Boundary polygon valid
- **Room Overlap** [HARD/PASS] — **10.0/10** (Excellent): No significant overlaps
- **Boundary Containment** [HARD/PASS] — **10.0/10** (Excellent): All rooms inside boundary
- **Minimum Area** [HARD/PASS] — **10.0/10** (Excellent): Areas OK
- **Minimum Width** [HARD/PASS] — **10.0/10** (Excellent): Widths OK
- **Reachability** [HARD/FAIL] — **0.0/10** (Poor): Unreachable: Room 1, Room 2, Room 1 Bath, Room 2 Bath, Kitchen, Dining
  - Suggestion: Add doors connecting unreachable rooms to circulation
- **Required Rooms Exist** [HARD/PASS] — **10.0/10** (Excellent): Required rooms present
- **Door Lines** [HARD/PASS] — **10.0/10** (Excellent): Doors have room connections
- **Window Lines** [HARD/FAIL] — **0.0/10** (Poor): Missing exterior glazing: ['Room 1', 'Room 2', 'Dining']
  - Suggestion: Add window line segments on exterior edges of living/bedrooms
- **Bathroom Accessible** [HARD/FAIL] — **0.0/10** (Poor): Bathroom lacks door connections
  - Suggestion: Connect bathroom to hall or bedroom
- **Kitchen Accessible** [HARD/FAIL] — **0.0/10** (Poor): Kitchen not connected
  - Suggestion: Connect kitchen to living/dining
- **Living Size** [soft/PASS] — **8.2/10** (Good): Living area 180.0 ft²
- **Bedroom Ratio** [soft/PASS] — **6.5/10** (Fair): Bedroom/living area ratio 1.99
  - Suggestion: Rebalance bedroom vs living areas
- **Natural Light** [soft/PASS] — **5.3/10** (Weak): Living: win 9.0/4.5 ft, WFR 22%, ext 12.0 ft, clear; Room 1: win 0.0/4.3 ft, WFR 0%, ext 12.0 ft, clear; Room 2: win 0.0/4.6 ft, WFR 0%, ext 12.0 ft, clear; Dining: win 0.0/3.5 ft, WFR 0%, ext 20.0 ft, clear
  - Suggestion: Add adequately sized windows on exterior walls of bedrooms/living
- **Privacy** [soft/PASS] — **7.5/10** (Good): Direct public-private adjacencies: 1
- **Circulation Efficiency** [soft/PASS] — **8.3/10** (Good): Circ 5.1%; min path width 12.0 ft; 0/2 beds via hall; 0 passageway rooms; 0 dead ends
- **Wet Core Compactness** [soft/FAIL] — **2.0/10** (Poor): Avg wet-room distance 16.56 ft
  - Suggestion: Cluster wet rooms for shared plumbing
- **Construction Shape** [soft/PASS] — **6.6/10** (Fair): Avg rectangularity 0.90; 0 short door edges
  - Suggestion: Simplify room shapes and place doors on longer shared walls
- **Room Proportions** [soft/PASS] — **10.0/10** (Excellent): Over-elongated rooms: 0
- **Storage Adequacy** [soft/PASS] — **4.0/10** (Weak): 0 storage spaces for 2 bedrooms
  - Suggestion: Attach private closets to their bedroom; keep shared closets near bedrooms
- **Balcony Usability** [soft/PASS] — **10.0/10** (Excellent): Balcony 90.0 ft²
- **Living Centrality** [soft/PASS] — **6.5/10** (Fair): Living 12.5 ft from plan center; mean 1.5 hops to other rooms
  - Suggestion: Place living nearer the center and connect rooms via hall/living hub
- **Kitchen–Public Link** [soft/FAIL] — **2.0/10** (Poor): Kitchen unreachable from living/dining
  - Suggestion: Connect kitchen to dining/living; never directly to a bedroom
- **Entry Sequence** [soft/PASS] — **7.5/10** (Good): no entry; living at exterior
- **Public Near Exit** [soft/PASS] — **9.5/10** (Excellent): Living 0 hops from exit (no bedrooms)

### Areas to improve
- Wet Core Compactness: Avg wet-room distance 16.56 ft → Cluster wet rooms for shared plumbing
- Kitchen–Public Link: Kitchen unreachable from living/dining → Connect kitchen to dining/living; never directly to a bedroom
- Storage Adequacy: 0 storage spaces for 2 bedrooms → Attach private closets to their bedroom; keep shared closets near bedrooms
- Natural Light: Living: win 9.0/4.5 ft, WFR 22%, ext 12.0 ft, clear; Room 1: win 0.0/4.3 ft, WFR 0%, ext 12.0 ft, clear; Room 2: win 0.0/4.6 ft, WFR 0%, ext 12.0 ft, clear; Dining: win 0.0/3.5 ft, WFR 0%, ext 20.0 ft, clear → Add adequately sized windows on exterior walls of bedrooms/living
- Bedroom Ratio: Bedroom/living area ratio 1.99 → Rebalance bedroom vs living areas
- Living Centrality: Living 12.5 ft from plan center; mean 1.5 hops to other rooms → Place living nearer the center and connect rooms via hall/living hub
- Construction Shape: Avg rectangularity 0.90; 0 short door edges → Simplify room shapes and place doors on longer shared walls

_Summary: This floor plan scores **2.72/10 (Poor)** for “3-room” with 10 rooms, 1176.0 ft² total room area. It fails 4 hard rule(s): Reachability, Window Lines, Bathroom Accessible, Kitchen Accessible. Strong areas include living size, circulation efficiency, room proportions. Weaker areas include natural light, wet core compactness, storage adequacy. Highest category is geometry (10.0); lowest is daylight (2.7)._