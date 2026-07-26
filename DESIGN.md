---
name: FaceForge AI
colors:
  surface: '#131318'
  surface-dim: '#131318'
  surface-bright: '#39393f'
  surface-container-lowest: '#0d0e13'
  surface-container-low: '#1b1b21'
  surface-container: '#1f1f25'
  surface-container-high: '#29292f'
  surface-container-highest: '#34343a'
  on-surface: '#e4e1e9'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#e4e1e9'
  inverse-on-surface: '#303036'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#c8c5ca'
  on-secondary: '#303033'
  secondary-container: '#47464a'
  on-secondary-container: '#b7b4b8'
  tertiary: '#c8c5cb'
  on-tertiary: '#303034'
  tertiary-container: '#919095'
  on-tertiary-container: '#29292d'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#e4e1e6'
  secondary-fixed-dim: '#c8c5ca'
  on-secondary-fixed: '#1b1b1e'
  on-secondary-fixed-variant: '#47464a'
  tertiary-fixed: '#e4e1e7'
  tertiary-fixed-dim: '#c8c5cb'
  on-tertiary-fixed: '#1b1b1f'
  on-tertiary-fixed-variant: '#47464b'
  background: '#131318'
  on-background: '#e4e1e9'
  surface-variant: '#34343a'
typography:
  display:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.03em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
    letterSpacing: -0.02em
  body-lg:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: '0'
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: '0'
  label-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1'
    letterSpacing: 0.05em
  mono-sm:
    fontFamily: Geist Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: '0'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  container-padding: 32px
  gutter: 24px
  grid-gap: 16px
  section-margin: 64px
---

## Brand & Style
The design system is anchored in an "AI Laboratory" aesthetic—a sophisticated, cinematic environment that balances technical precision with high-end luxury. The interface evokes the feeling of a controlled experimental space where synthetic intelligence is crafted.

The style leverages **Glassmorphism** and **Modern Minimalism**. It utilizes deep obsidian surfaces, semi-transparent layers, and ethereal glows to create a sense of depth and advanced technology. The emotional response should be one of quiet power, intelligence, and absolute trust in the generative process.

## Colors
This design system employs a "Dark-First" hierarchy to minimize ocular strain during long generation sessions and to emphasize the vibrance of synthetic outputs.

- **Primary (Electric Violet):** Reserved for active states, primary actions, and "live" generation indicators.
- **Surface Tiers:**
    - **Base:** Deep Charcoal (#0F0F12) for the canvas.
    - **Container:** Obsidian (#1A1A1E) for primary UI sections.
    - **Elevated:** Slate (#2C2C32) for floating panels and tooltips.
- **Interactive Effects:** Use the primary color with a soft Gaussian blur to create a "generation glow" effect around active AI nodes.

## Typography
The typography system uses **Geist** to lean into a developer-centric yet premium feel.

- **Headings:** Use tight tracking (negative letter-spacing) to create a high-fashion, cinematic impact.
- **Body:** Use generous line-heights (1.6) to maintain legibility against dark backgrounds.
- **Data Labels:** Use uppercase tracking for metadata, such as "Seed Number" or "Latent Dimension," to evoke laboratory instrumentation.

## Layout & Spacing
The layout follows a **Fluid Grid** model with high internal margins to emphasize the "Cinematic" direction.

- **Synthetic Face Grid:** Faces should be presented in a responsive grid using a `16px` gap. Use a 1:1 aspect ratio for all generation previews.
- **Training Charts:** Use a fixed-height container for loss charts to ensure temporal data is readable without vertical scrolling.
- **Desktop:** 12-column grid with `32px` side margins.
- **Mobile:** Single column with `20px` margins; reduce headline sizes as specified in the typography tokens.

## Elevation & Depth
Depth is communicated through **Glassmorphism** rather than traditional shadow stacking.

- **Surface Treatment:** All primary panels use a background blur (20px - 40px) and a semi-transparent fill (`#1A1A1E80`).
- **Borders:** Surfaces are defined by a `1px` solid border (`#FFFFFF1A`). This "internal stroke" mimics the appearance of glass edges.
- **Shadows:** Use a single, large-radius ambient shadow (`0 20px 40px rgba(0,0,0,0.4)`) for floating modals to separate them from the base laboratory floor.
- **Glows:** Primary buttons and active generation nodes should emit a soft, `8px` outer glow in the primary violet color.

## Shapes
The shape language is modern and approachable.

- **Standard Containers:** Use `16px` (rounded-lg) for main cards, panels, and face previews.
- **Interactive Elements:** Buttons and input fields should follow the `8px` (rounded-md) standard to maintain a crisp, professional look.
- **Selection States:** Use a `2px` primary violet border to indicate selected faces in the grid.

## Components

### Buttons
- **Primary:** Solid `#8B5CF6` with white text. High-gloss finish.
- **Secondary:** Transparent background with the `1px` border token and a hover state that increases background opacity.

### Synthetic Face Grid
- Cards should be borderless by default, relying on the grid gap.
- On hover, an overlay should appear showing metadata (Seed, CFG Scale) using the `mono-sm` typography style.

### Input Fields
- Dark backgrounds (`#0F0F12`) with the `1px` border.
- Focus state: Border color changes to the primary violet with a subtle outer glow.

### Training Loss Charts
- Use a monochromatic slate color for the grid lines and the primary violet for the data line.
- Background of the chart should be slightly darker than the surrounding panel to create a "recessed" look.

### Progress Indicators
- For face generation, use a "shimmer" effect across the glass container rather than a standard bar. This reinforces the "experimental" brand vibe.
