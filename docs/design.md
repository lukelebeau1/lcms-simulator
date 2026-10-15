# Design: the control-room look

The app is styled after the main control room of a 1970s Soviet nuclear plant (inspiration only: the Leningrad
plant's control room photo, RIA Novosti archive 305015, Alexey Danichev, CC BY-SA 3.0 on Wikimedia Commons; the
photo is not part of this repository). The forms come from the room: a title band, numbered panel sections, an
annunciator for the "simulated" warning, readout tiles, push-buttons, and backlit tiles on the station rail.

## Where it lives

- `.streamlit/config.toml`, theme section: page colours, fonts, square corners.
- `app/style.py`: what the theme cannot express (title band, section strips, annunciator, tiles, rail) as page CSS,
  put in by `common.page()`; and the plotly template `room`, the default for every chart.
- `common.chart()` and `common.plotly_spec()` draw charts with `theme=None` and set the screen colour on the figure,
  because Streamlit otherwise fills unset backgrounds from the page theme.

## Palette

| Role | Colour |
|---|---|
| Panels | sand `#E7DFD0`, sections `#DED4C2`, seams `#A89C86`, sidebar `#CDC4B6` |
| Ink | violet-black `#2E2433`, muted `#5C5262` |
| System, pressed controls | deep violet `#574066`, lilac `#BA9DDB`, pale lilac `#EDE5F5` |
| Title band, labels, attention, high | saffron `#D9A441`, pale saffron `#F3DFA8` |
| On target | cream `#F3ECDD` |
| Low | teal `#3F8A8C` (light `#7CC4C6` on screens) |
| Chart screens | deep violet `#3A2C48`, grid `#574066` |

Marks on a screen go lilac, saffron, light teal, cream, pale lilac. Mid violet stays off the screens: it merges with
teal under deuteranopia.

## Type

PT Sans Narrow for text and headings, PT Mono for numbers, readouts and the warning (Google Fonts, loaded by the theme).
