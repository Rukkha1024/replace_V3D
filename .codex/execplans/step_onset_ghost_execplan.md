# live 모드 step_onset 고스트 스냅샷 / Live Mode step_onset Ghost Snapshot

This ExecPlan is a living document. Update `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` as work proceeds.

Maintained per `.claude/PLANS.md`.

---

## Purpose / Big Picture

**EN:** In live-mode GIFs from `scripts/plot_bos_com_xy_sample.py`, the BOS rectangle
continuously updates each frame. When the animation plays past `step_onset`, there is no
visual record of what the BOS and COM state looked like at that exact moment. The user
wants a persistent "ghost snapshot" overlaid on the live animation: a semi-transparent
dashed rectangle showing the BOS bounds at `step_onset`, plus a distinctive marker showing
the COM position at `step_onset`. Both remain visible from the onset frame onward so the
viewer can compare "what it was right then" against the current frame. A small text label
annotates the ghost COM point with the step_onset frame number.

Success: open a live-mode GIF; from the step_onset frame onward a dashed orange BoS ghost
rectangle and a diamond COM ghost marker appear and persist to the end of the clip.

**KO:** `scripts/plot_bos_com_xy_sample.py`의 live 모드 GIF에서 BOS 사각형이 매 프레임
업데이트되므로, step_onset 시점의 BoS·COM 상태가 이후 프레임에 남지 않는다. 사용자는
해당 시점의 BoS 경계를 반투명 점선 사각형으로, COM 위치를 다이아몬드 마커로 고정
표시하여(이후 프레임에서도 유지) "그 순간 어땠는지"를 동시에 확인하고 싶다.
라벨로 step_onset 프레임 번호도 표시한다.

---

## Progress

- [ ] ExecPlan drafted and approved by user.
- [ ] ExecPlan written to `.codex/execplans/step_onset_ghost_execplan.md`.
- [ ] Implement ghost setup block in `render_gif()`.
- [ ] Implement per-frame ghost update in `update()`.
- [ ] Add ghost artists to blit list.
- [ ] Run `--step_vis phase_bos --all --jobs 2` and visually verify ghost appears at step_onset.
- [ ] Git commit (Korean, ≥3 lines).

---

## Surprises & Discoveries

(Fill in during implementation.)

---

## Decision Log

- Decision: Ghost feature is always-on for live mode when `step_onset_idx` is found; no separate
  CLI flag needed.
  Rationale: The user described this as always desired in live mode — a toggle flag adds
  complexity without benefit since `step_vis none` already turns off all step-related overlays.
  Date/Author: 2026-02-18 / Claude

- Decision: Ghost elements are hidden before `step_onset_idx` and become visible at/after it.
  Rationale: Showing the ghost before the onset event would be misleading (that state hasn't
  happened yet). The ghost appears exactly at onset and persists — matching "아 이때 이랬구나".
  Date/Author: 2026-02-18 / Claude

- Decision: Ghost applies only in `live` BOS mode (not `freeze`).
  Rationale: In `freeze` mode the live BOS rect itself freezes at step_onset, making a
  separate ghost rectangle redundant and visually confusing.
  Date/Author: 2026-02-18 / Claude

---

## Outcomes & Retrospective

(Fill in after implementation.)

---

## Context and Orientation

**Primary file:** `scripts/plot_bos_com_xy_sample.py`

Key zones inside `render_gif()`:

| Zone | Lines (approx) | Role |
|------|---------------|------|
| `bos_freeze_idx` computation | 1105–1121 | freeze-mode BOS lock |
| `step_vis` setup block | 1178–1201 | template artist setup |
| `update()` function | 1237–1330 | per-frame rendering |
| `bos_phase/phase_bos` color flash | 1300–1318 | step_vis color logic |
| Artists blit list | 1321–1330 | returned artist tuple |

`step_onset_idx` (L1180–1189) is already computed independently of freeze mode — we reuse
it directly.

`display.com_x[step_onset_idx]` / `display.bos_minx[step_onset_idx]` etc. give the snapshot
values we need.

---

## Plan of Work

### Section 1 — Ghost artist setup (after existing `step_vis` setup block, ~L1201)

```python
# ---- ghost snapshot setup (live mode only) ----
ghost_bos_rect: Rectangle | None = None
ghost_com_pt = None
ghost_label = None
if mode == "live" and step_onset_idx is not None:
    ghost_bos_rect = Rectangle(
        (0.0, 0.0), width=0.0, height=0.0,
        facecolor="none",
        edgecolor="darkorange",
        alpha=0.0,          # hidden until step_onset reached
        linewidth=2.0,
        linestyle="--",
        zorder=6,
    )
    ax.add_patch(ghost_bos_rect)
    (ghost_com_pt,) = ax.plot(
        [], [],
        marker="D",         # diamond
        linestyle="None",
        markersize=9,
        markerfacecolor="darkorange",
        markeredgecolor="black",
        markeredgewidth=0.8,
        alpha=0.0,
        zorder=7,
    )
    ghost_label = ax.text(
        0.0, 0.0, "",
        fontsize=6.5,
        color="darkorange",
        ha="left", va="bottom",
        zorder=8,
        visible=False,
    )
# ---- end ghost setup ----
```

### Section 2 — Per-frame ghost update (inside `update()`, after step_vis color block, ~L1319)

```python
# ---- ghost snapshot per-frame ----
if ghost_bos_rect is not None and ghost_com_pt is not None and step_onset_idx is not None:
    if idx >= step_onset_idx:
        # Snapshot values captured once at step_onset_idx
        g_minx = float(display.bos_minx[step_onset_idx])
        g_maxx = float(display.bos_maxx[step_onset_idx])
        g_miny = float(display.bos_miny[step_onset_idx])
        g_maxy = float(display.bos_maxy[step_onset_idx])
        ghost_bos_rect.set_xy((g_minx, g_miny))
        ghost_bos_rect.set_width(g_maxx - g_minx)
        ghost_bos_rect.set_height(g_maxy - g_miny)
        ghost_bos_rect.set_alpha(0.75)

        g_cx = float(display.com_x[step_onset_idx])
        g_cy = float(display.com_y[step_onset_idx])
        ghost_com_pt.set_data([g_cx], [g_cy])
        ghost_com_pt.set_alpha(1.0)

        if ghost_label is not None:
            ghost_label.set_position((g_cx + 0.01, g_cy + 0.01))
            ghost_label.set_text(f"step@{int(series.mocap_frame[step_onset_idx])}")
            ghost_label.set_visible(True)
    else:
        ghost_bos_rect.set_alpha(0.0)
        ghost_com_pt.set_alpha(0.0)
        if ghost_label is not None:
            ghost_label.set_visible(False)
# ---- end ghost per-frame ----
```

### Section 3 — Artists blit list (inside `update()`, ~L1321)

Extend the `artists` list with ghost elements:

```python
if ghost_bos_rect is not None:
    artists.append(ghost_bos_rect)
if ghost_com_pt is not None:
    artists.append(ghost_com_pt)
if ghost_label is not None:
    artists.append(ghost_label)
```

---

## Concrete Steps

1. Read `scripts/plot_bos_com_xy_sample.py` in full to pinpoint exact insertion lines.
2. Insert **ghost setup block** after the `# ---- end step_vis setup ----` comment (currently L1201).
3. Insert **ghost per-frame update** after the `# ---- end step_vis per-frame ----` comment (currently L1319), inside `update()`.
4. Extend **artists list** (currently L1321–1330) with ghost elements.
5. Run:
   ```bash
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n module python \
     scripts/plot_bos_com_xy_sample.py --step_vis phase_bos --jobs 2
   ```
6. Open the resulting `*__step_vis-phase_bos__live.gif` and verify:
   - Before step_onset: no ghost elements.
   - At step_onset frame: dashed orange BOS ghost + orange diamond COM ghost appear.
   - After step_onset: ghost elements remain stable while live BOS continues to update.
7. Also verify `--step_vis none` (freeze+live) GIFs are byte-identical to previous outputs
   (ghost only activates when `mode == "live" and step_onset_idx is not None`; no effect on
   `step_vis none` path since ghosts are not created there — the guard `if mode == "live"`
   ensures this).
8. Git commit (Korean, ≥3 lines).
9. Copy final plan to `.codex/execplans/step_onset_ghost_execplan.md`.

---

## Validation and Acceptance

| Check | Method | Expected |
|-------|--------|---------|
| Ghost appears at onset | Open `*__step_vis-phase_bos__live.gif` | Dashed orange BOS rect + diamond marker visible from step_onset frame |
| Ghost persists after onset | Scroll GIF to end | Ghost BOS and ghost COM remain in same fixed position |
| Ghost absent before onset | Scroll GIF to frame before onset | No ghost elements visible |
| No regression on freeze | Open `*__right1col__freeze.gif` | Identical to previous (no ghost in freeze mode) |
| No regression on `step_vis none` | Open `*__right1col__live.gif` | Identical to previous (no ghost elements) |

---

## Idempotence and Recovery

- Ghost elements are purely additive artists; they do not modify existing artist state.
- If `step_onset_idx is None`, ghost setup block is skipped entirely → no change in behavior.
- If `mode == "freeze"`, ghost setup is skipped → no change for freeze-mode GIFs.
- Re-running the script always overwrites output GIFs (existing `out_path.unlink()` at L1344).

---

## Artifacts and Notes

- **Modified file:** `scripts/plot_bos_com_xy_sample.py`
- **New execplan:** `.codex/execplans/step_onset_ghost_execplan.md`
- Ghost label uses a small 6.5pt offset to avoid overlap with the diamond marker.
- `blit=False` is already set in `FuncAnimation` (L1341), so `Text` artists are safe to include
  in the returned tuple without the blit-incompatibility issue.

---

## Interfaces and Dependencies

- `display.com_x`, `display.com_y`: COM trajectory arrays (already used)
- `display.bos_minx`, `display.bos_maxx`, `display.bos_miny`, `display.bos_maxy`: BOS bounds (already used)
- `series.mocap_frame[step_onset_idx]`: frame number for ghost label
- No new imports needed (`Rectangle` already imported, `matplotlib` artists already available)
