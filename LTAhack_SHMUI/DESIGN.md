---
name: LTA Rail Sentinel
colors:
  surface: '#f7fafe'
  surface-dim: '#d7dade'
  surface-bright: '#f7fafe'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f1f4f8'
  surface-container: '#ebeef2'
  surface-container-high: '#e5e8ec'
  surface-container-highest: '#e0e3e7'
  on-surface: '#181c1f'
  on-surface-variant: '#45474b'
  inverse-surface: '#2d3134'
  inverse-on-surface: '#eef1f5'
  outline: '#76777b'
  outline-variant: '#c6c6cb'
  surface-tint: '#5d5e63'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#1a1c20'
  on-primary-container: '#828389'
  inverse-primary: '#c6c6cc'
  secondary: '#735c00'
  on-secondary: '#ffffff'
  secondary-container: '#fdcc00'
  on-secondary-container: '#6e5700'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#00210a'
  on-tertiary-container: '#079847'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2e2e8'
  primary-fixed-dim: '#c6c6cc'
  on-primary-fixed: '#1a1c20'
  on-primary-fixed-variant: '#45474b'
  secondary-fixed: '#ffe086'
  secondary-fixed-dim: '#efc100'
  on-secondary-fixed: '#231a00'
  on-secondary-fixed-variant: '#574500'
  tertiary-fixed: '#83fb9d'
  tertiary-fixed-dim: '#66de83'
  on-tertiary-fixed: '#00210a'
  on-tertiary-fixed-variant: '#005323'
  background: '#f7fafe'
  on-background: '#181c1f'
  surface-variant: '#e0e3e7'
typography:
  display-lg:
    fontFamily: Public Sans
    fontSize: 40px
    fontWeight: '800'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl:
    fontFamily: Public Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.01em
  headline-xl-mobile:
    fontFamily: Public Sans
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Public Sans
    fontSize: 22px
    fontWeight: '700'
    lineHeight: 28px
  headline-sm:
    fontFamily: Public Sans
    fontSize: 16px
    fontWeight: '700'
    lineHeight: 22px
    letterSpacing: 0.02em
  body-lg:
    fontFamily: Public Sans
    fontSize: 15px
    fontWeight: '500'
    lineHeight: 22px
  body-md:
    fontFamily: Public Sans
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-mobile: 0.75rem
  margin: 1.5rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system translates Singapore's Land Transport Authority (LTA) physical signage and architectural wayfinding standards into a high-density, mission-critical predictive maintenance interface. Designed for operations controllers, rolling stock technicians, and track maintenance engineers, the aesthetic establishes instant operational legibility, strict structural authority, and zero-latency situational awareness.

The design movement combines **Transit Functionalism** with **Industrial Brutalism**:
- **Wayfinding Rigor**: Layouts borrow directly from Singapore MRT station wayfinding totems, overhead signage bands, and track interchange diagrams.
- **Architectural Clarity**: Heavy headers, high-contrast badges, crisp mechanical keylines, and strictly demarcated visual lanes enforce hierarchy under high-stress conditions.
- **Instrumental Metaphor**: Health states reject soft, abstract gradients in favor of calibrated segment arrays, segmented bar batteries, and discrete structural status indicators derived from railway instrumentation panels.

## Colors

The palette is derived from Singapore's physical rapid transit identity, anchoring critical operational statuses to universally recognized line colors and signage conventions.

### Core System Tokens
- **Control Black (`#0C0E12`)**: The definitive header, structural masthead, and terminal surface tone. Used for system-critical navbars and anchoring bands.
- **Wayfinding Yellow (`#FFCE00`)**: Derived from the signature LTA station "EXIT" archway signage. Reserved for critical terminal exits, overrides, primary actions, and cautionary alerts.
- **East-West Green (`#009645`)**: Used for optimal equipment health, nominal system states, completed inspection tracks, and cleared work orders.
- **North-South Red (`#D42E12`)**: High-priority alert state, critical vibration threshold breaches, structural fractures, and urgent stoppages.
- **Circle Line Amber (`#FF9900`)**: Predictive degradation warnings, maintenance required within 48 hours, and thermal thresholds.
- **Downtown Blue (`#005EC4`)**: Telemetry signals, diagnostic sensor reads, electrical subsystem indicators, and secondary navigation pills.
- **Transit Base (`#F1F4F8`)**: Primary canvas tone resembling low-glare station wall cladding.
- **Surface Elevation (`#E8EDF3`)**: Sub-panel surface and subtle repeating architectural grid/dot-matrix pattern backdrop.
- **Keyline Border (`#D1D9E2`)**: Structural 1px border that frames modular telemetry panels.

## Typography

The type system prioritizes uncompromised legibility and exact data alignment:

- **Signage Sans (`Public Sans`)**: Closely mirrors the geometric, authoritative humanist sans-serif standard deployed across LTA station platforms and directional overheads. Used for section titles, line codes, station names, and modal actions.
- **Instrument Monospace (`JetBrains Mono`)**: Strictly applied to component serial numbers, bearing vibration metrics (Hz/mm/s), bogie temperature outputs, run-hour totals, and track chainage coordinates (`CH: 14+230`).
- **Casing & Kerning**: Primary station indicators, line designations (e.g., `EW24`, `NS1`, `CC22`), and maintenance state triggers are formatted in uppercase with slight tracking expansion to simulate etched metal signage plates.

## Layout & Spacing

The layout is anchored in a high-density, multi-panel structural grid designed for operations monitors, rugged field tablets, and dispatch consoles.

- **Canvas Foundation**: The backdrop uses an architectural subtle grid pattern (16px grid lines in `#E8EDF3` over `#F1F4F8`) evoking civil engineering schematics and station tile architecture.
- **Grid Structure**:
  - **Desktop (1440px+)**: 12-column rigid grid with a permanent 64px black top navigation band and fixed 320px telemetry inspection rail.
  - **Tablet (768px - 1023px)**: 8-column layout; secondary telemetry drawers collapse into docked slide-over sheets.
  - **Mobile (< 768px)**: Single column with horizontal swipeable card stacks for bogie and bogie-wheelset groups.
- **Spacing Rhythm**: Employs an exact 4px/8px modular cadence (`0.25rem` to `2rem`). Padding within telemetry cards remains compact to maximize above-the-fold telemetry density.

## Elevation & Depth

This system avoids blurred drop shadows and skeuomorphic lighting, utilizing **Architectural Layering and Crisp Line Contours** instead:

- **Zero-Shadow Rule**: Elevation is communicated entirely through solid contrasting surface values, 1px keylines, and intentional border color changes.
- **Masthead Layer (`#0C0E12`)**: Pure deep black header sitting at the highest visual plane, bordered beneath by a 3px solid multi-color line ribbon (EW Green, NS Red, CC Amber, DT Blue).
- **Panel Base Layer (`#FFFFFF`)**: Pure white data tiles resting directly over `#F1F4F8` with a mandatory `1px solid #D1D9E2` stroke.
- **Active / Alert Elevation**: When a component triggers a warning or critical threshold, elevation is communicated by intensifying the border to `2px solid #D42E12` or `#FF9900` along with a subtle tinted background wash (`#D42E12` at 4% opacity).
- **Sub-Component Recesses**: Sensor input containers and telemetry logs sit within a 1-step recessed container (`#E8EDF3`) with `inset 0 1px 2px rgba(12, 14, 18, 0.06)`.

## Shapes

The design embraces a structural, mechanical corner radius of **0.25rem (4px)** for cards, buttons, and telemetry cells, matching the industrial cut of route maps and rail electrical housing boxes.

- **Standard Cards & Modals**: 4px radius (`roundedness: 1`), keeping lines rigid and precision-engineered.
- **Station Badges & Line Tokens**: Standardized as rounded rectangles (4px) with high-contrast centered text (e.g., green fill for `[EW]`, red for `[NS]`).
- **Signature Wayfinding EXIT Element**: Utilizes an inverted U-shape / open-bottom archway (`border-radius: 6px 6px 0 0`), mirroring physical Singapore MRT egress signage.

## Components

### 1. The Signature LTA EXIT Button
- **Structure**: Modeled after the iconic station overhead "EXIT" signage.
- **Visual Spec**: Solid `#FFCE00` background, high-contrast `#0C0E12` bold sans-serif text, encased in a distinctive open-bottom arch: `border-radius: 6px 6px 0 0` with a 2px solid `#0C0E12` perimeter on top, left, and right, but left open/flush along the bottom edge.
- **Role**: Emergency isolation triggers, maintenance window cut-offs, or active system logout/egress.

### 2. Battery & Structural Health Gauges
- **Segmented Health Bar**: Consists of 10 discrete vertical segments (each 4px wide, 16px high, separated by a 2px gap).
  - Segments 1–3: `#009645` (Nominal / Good)
  - Segments 4–7: `#FF9900` (Elevated Wear)
  - Segments 8–10: `#D42E12` (Critical Maintenance Breach)
- **Degradation Ring Gauge**: Monospaced value centered inside an open SVG radial track (270° sweep), featuring mechanical tick marks around the perimeter.

### 3. Transit Line Badges & Filter Chips
- **Line Badges**: Rectangular capsules with solid background fills corresponding directly to line codes:
  - `NSL`: `#D42E12` background, white text.
  - `EWL`: `#009645` background, white text.
  - `CCL`: `#FF9900` background, `#0C0E12` text.
  - `DTL`: `#005EC4` background, white text.
- **Filter Chips**: Neutral gray (`#E8EDF3`) default, activating into high-contrast inverted black (`#0C0E12`) with an underline matching the selected transit line color.

### 4. Telemetry Metric Cards
- **Header**: Micro-label in `JetBrains Mono` displaying component reference tag (`BOGIE-TR2-AXLE-04`), right-aligned uptime timestamp.
- **Main Metric**: Giant tabular number (e.g., `14.2 mm/s`) accompanied by trend-indicator arrows.
- **Frame**: 1px solid `#D1D9E2`, white card background, no dropshadow.

### 5. Input Fields & Search Bars
- **Style**: High-visibility industrial input boxes with a 1px solid `#D1D9E2` border, sharp 4px corners, and monospaced placeholder text. Focus state transforms border to 2px solid `#005EC4` with zero external ring blur.

### 6. Lists & Log Tables
- **Row Styling**: Alternating hairline separators (`#E8EDF3`), dense 36px row heights, monospaced timestamp and error code columns, with a 3px thick color-coded left border indicating fault severity.