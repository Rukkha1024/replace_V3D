# step_onset 시각화 템플릿 추가 / Add step_onset Visualization Templates

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `.claude/PLANS.md` from the repository root.

## Purpose / Big Picture

EN: Currently `step_onset_local` is surfaced only as a one-frame text label and a BOS-freeze trigger
inside `scripts/plot_bos_com_xy_sample.py`. After this change the user can run
`python scripts/plot_bos_com_xy_sample.py --step_vis all` and receive 8 GIF files (4 visual
templates × 2 BOS modes: freeze/live) for the selected trial. Each GIF makes it immediately
apparent when the step occurs using a different visual encoding. The user watches all 8 GIFs
and picks the style they prefer; that style becomes the default going forward (kept via `--step_vis`).
Success is visible: open any `*__step_vis-*__freeze.gif`, see a gold star at the step-onset position,
a timeline bar in the right panel with an orange vertical line at step onset, and (depending on
template) color-coded COM trail or BOS phase change.

KO: 현재 `step_onset_local`은 `scripts/plot_bos_com_xy_sample.py` 안에서 한 프레임짜리 텍스트
레이블과 BOS 고정 트리거로만 표시된다. 이 변경 후 사용자는
`python scripts/plot_bos_com_xy_sample.py --step_vis all` 을 실행하면 선택 trial에 대해
GIF 8개 (시각화 템플릿 4종 × BOS 모드 2종: freeze/live)를 받는다. 각 GIF는 서로 다른 시각
인코딩으로 step이 언제 발생했는지를 즉각적으로 표시한다. 사용자가 8개를 보고 선호하는 스타일을
선택하면, 그 스타일을 `--step_vis` 인자로 고정 사용한다.
성공 확인: `*__step_vis-*__freeze.gif` 열어서 금색 별 마커, 우측 패널 타임라인 바 + 주황 수직선,
(템플릿에 따라) COM 트레일 색상 분리 또는 BOS 페이즈 색상 변화가 보이면 성공이다.

## Progress

- [x] (2026-02-18) ExecPlan drafted and approved by user.
- [x] (2026-02-18) ExecPlan written to `.codex/execplans/step_onset_vis_execplan.md`.
- [x] (2026-02-18) Read full `scripts/plot_bos_com_xy_sample.py` for precise code structure.
- [x] (2026-02-18) Added `STEP_VIS_TEMPLATES` constant, `step_vis` field to `RenderConfig`, `--step_vis` CLI arg.
- [x] (2026-02-18) Implemented `add_timeline_inset()` helper function.
- [x] (2026-02-18) Modified `render_gif()` to accept `step_vis` and implement 4 templates + timeline.
- [x] (2026-02-18) Added `STEP_VIS_ALL` dispatch in `render_one_trial()`.
- [x] (2026-02-18) Ran `--step_vis all` → 8 GIFs produced for trial 가윤호-60-1.
- [x] (2026-02-18) Ran `--step_vis none` → original `*__right1col__freeze/live.gif` filenames preserved.
- [ ] Git commit with Korean message.

## Surprises & Discoveries

(To be filled in during implementation.)

## Decision Log

- Decision: Render all 4 templates with `--step_vis all` rather than via CLI flag per-template-only.
  Rationale: User requested "render all and compare"; a single run producing comparison set is
  the fastest path to the user's selection decision.
  Date/Author: 2026-02-18 / Claude

- Decision: Timeline bar uses `ax_side.inset_axes()` rather than a new subplot row.
  Rationale: Adding a new subplot row would change the overall figure geometry and break
  existing aspect-ratio and legend sizing. An inset within the existing right panel
  is additive and does not affect the main XY plot geometry.
  Date/Author: 2026-02-18 / Claude

- Decision: `--step_vis none` (default) leaves existing behavior byte-identical.
  Rationale: Backwards compatibility — existing batch runs without `--step_vis` must continue
  to produce the same GIF output so MD5 comparisons in QC still pass.
  Date/Author: 2026-02-18 / Claude

## Outcomes & Retrospective

EN: Implementation complete. Added `--step_vis` CLI flag with 4 template styles + `all` mode.
`--step_vis all` produces 8 GIFs (4 templates × 2 BOS modes) for the selected trial. Timeline
inset appears in right panel for all non-none templates. `--step_vis none` (default) preserves
exact original filenames and behavior. Trial 가윤호-60-1 used for validation (223 frames,
step_onset_local=126). The conda `UnicodeEncodeError` on Windows with Korean paths is a
pre-existing conda-level issue; the Python script itself succeeds (all GIFs saved).

KO: 구현 완료. `--step_vis` CLI 플래그 4가지 템플릿 + `all` 모드 추가.
`--step_vis all` 실행 시 선택 trial에 GIF 8개(4 템플릿 × 2 BOS 모드) 생성.
모든 non-none 템플릿에서 우측 패널 타임라인 인셋 표시됨.
`--step_vis none`(기본값)은 기존 파일명과 동작을 완전히 보존.
검증 trial: 가윤호-60-1 (223 프레임, step_onset_local=126).
Windows 한국어 경로의 conda `UnicodeEncodeError`는 기존 conda 수준 이슈이며
Python 스크립트 자체는 정상 실행 (모든 GIF 저장 완료).

## Context and Orientation

EN: The sole entry script is `scripts/plot_bos_com_xy_sample.py`. It reads
`output/all_trials_timeseries.csv` (polars DataFrame), selects one `(subject, velocity, trial)`,
and renders GIF animations via matplotlib `FuncAnimation`. The two GIF modes are `freeze`
(BOS rectangle frozen at `step_onset_local`) and `live` (BOS tracks current frame). The
figure layout uses two matplotlib axes side by side: `ax` (XY plot, width ratio 3.45) and
`ax_side` (right info panel, width ratio 1.15). The right panel currently contains only a
`ax_side.text()` artist (`info_text`) updated each frame.

`step_onset_local` is stored in `TrialSeries.step_onset_local` (Optional[int]). The valid
index of that frame is computed around line 1054 as `bos_freeze_idx`. We reuse the same
index as `step_onset_idx` for all template rendering.

Key paths:
- Entry script: `scripts/plot_bos_com_xy_sample.py`
- Data: `output/all_trials_timeseries.csv`
- Output: `output/figures/bos_com_xy_sample/<subject>/`

KO: 유일한 진입 스크립트는 `scripts/plot_bos_com_xy_sample.py`다. `output/all_trials_timeseries.csv`
(polars DataFrame)를 읽어 `(subject, velocity, trial)` 하나를 선택하고 matplotlib `FuncAnimation`
으로 GIF를 렌더링한다. GIF 모드 두 가지: `freeze`(BOS 사각형이 `step_onset_local`에서 고정)와
`live`(BOS가 현재 프레임 추적). 피겨 레이아웃은 두 축 나란히: `ax`(XY 플롯, 너비 비율 3.45)와
`ax_side`(우측 정보 패널, 너비 비율 1.15). 우측 패널은 현재 `ax_side.text()` 아티스트(`info_text`)
하나만 포함하며 매 프레임 갱신된다.

`step_onset_local`은 `TrialSeries.step_onset_local` (Optional[int])에 저장된다. 해당 프레임의
valid 인덱스는 line 1054 부근에서 `bos_freeze_idx`로 계산된다. 모든 템플릿 렌더링에서 동일한
인덱스를 `step_onset_idx`로 재사용한다.

## Plan of Work

### 1. CLI argument and RenderConfig

In `build_arg_parser()` (search for `--bos_mode` to find the location), add:

    parser.add_argument(
        "--step_vis",
        default="none",
        choices=["none", "phase_trail", "bos_phase", "star_only", "phase_bos", "all"],
        help=(
            "Step-onset visualization style. "
            "'none' = current behavior (no change). "
            "'all' = render all 4 template styles for comparison."
        ),
    )

In `RenderConfig` dataclass, add field `step_vis: str = "none"`.

In `build_render_config()`, add `step_vis=str(args.step_vis)` to the constructor call.

### 2. Helper: add_timeline_inset()

Add a new module-level function after `apply_gif_right_panel()`:

    def add_timeline_inset(
        ax_side: plt.Axes,
        series: TrialSeries,
    ) -> matplotlib.lines.Line2D:
        """Add a horizontal timeline strip at the bottom of ax_side.

        Returns the animated cursor Line2D that must be updated each frame.
        """
        ax_tl = ax_side.inset_axes([0.04, 0.02, 0.92, 0.18])
        first_frame = int(series.mocap_frame[0])
        last_frame = int(series.mocap_frame[-1])
        ax_tl.set_xlim(first_frame, last_frame)
        ax_tl.set_ylim(0.0, 1.0)
        ax_tl.axis("off")
        # Base horizontal bar
        ax_tl.axhline(0.5, color="0.55", linewidth=2.0, solid_capstyle="round")
        # Platform onset/offset markers
        ax_tl.axvline(series.platform_onset_local, color="0.45", linewidth=1.2,
                      linestyle="--")
        ax_tl.axvline(series.platform_offset_local, color="0.45", linewidth=1.2,
                      linestyle="--")
        ax_tl.text(series.platform_onset_local, 0.15, "on", fontsize=5,
                   ha="center", color="0.45")
        ax_tl.text(series.platform_offset_local, 0.15, "off", fontsize=5,
                   ha="center", color="0.45")
        # Step onset marker (orange)
        if series.step_onset_local is not None:
            ax_tl.axvline(int(series.step_onset_local), color="tab:orange",
                          linewidth=2.2)
            ax_tl.text(int(series.step_onset_local), 0.82, "step",
                       fontsize=5, ha="center", color="tab:orange",
                       fontweight="bold")
        # Animated cursor
        cursor_line = ax_tl.axvline(first_frame, color="tab:blue", linewidth=1.8,
                                    alpha=0.85)
        return cursor_line

### 3. Modify render_gif_animation()

Add `step_vis: str = "none"` parameter to `render_gif_animation()`.

After `bos_freeze_idx` is resolved (line ~1069), alias it:
    step_onset_idx: int | None = bos_freeze_idx  # reuse same valid index

Then, before creating artists, branch on `step_vis`:

**Star marker (all templates except "none"):**

    step_star_artist = None
    if step_vis != "none" and step_onset_idx is not None:
        xs = float(display.com_x[step_onset_idx])
        ys = float(display.com_y[step_onset_idx])
        step_star_artist, = ax.plot(
            [xs], [ys],
            marker="*",
            linestyle="None",
            markersize=14,
            markerfacecolor="gold",
            markeredgecolor="darkorange",
            markeredgewidth=0.9,
            zorder=7,
        )

**Timeline inset (all templates except "none"):**

    timeline_cursor: matplotlib.lines.Line2D | None = None
    if step_vis != "none":
        timeline_cursor = add_timeline_inset(ax_side, series)

**Phase trail artists (phase_trail, phase_bos):**

    trail_pre = trail_post = None
    if step_vis in ("phase_trail", "phase_bos"):
        trail_line.set_visible(False)  # hide original single trail
        trail_pre, = ax.plot([], [], color="tab:blue", linewidth=2.0,
                             alpha=0.95, zorder=3)
        trail_post, = ax.plot([], [], color="tab:orange", linewidth=2.0,
                              alpha=0.95, zorder=3)

**In update() closure**, extend the existing logic:

    # Timeline cursor
    if timeline_cursor is not None:
        timeline_cursor.set_xdata([frame_value])

    # Phase trail split
    if trail_pre is not None and trail_post is not None:
        if step_onset_idx is not None:
            pre_hist = valid_indices[(valid_indices <= idx) & (valid_indices <= step_onset_idx)]
            post_hist = valid_indices[(valid_indices <= idx) & (valid_indices > step_onset_idx)]
        else:
            pre_hist = history
            post_hist = np.array([], dtype=int)
        trail_pre.set_data(display.com_x[pre_hist], display.com_y[pre_hist])
        trail_post.set_data(display.com_x[post_hist], display.com_y[post_hist])

    # BOS phase color (bos_phase, phase_bos)
    if step_vis in ("bos_phase", "phase_bos") and step_onset_idx is not None:
        if idx >= step_onset_idx:
            bos_rect.set_facecolor("lightyellow")
            bos_rect.set_edgecolor("tab:orange")
        else:
            bos_rect.set_facecolor("lightskyblue")
            bos_rect.set_edgecolor("tab:blue")

**Return value of update():** include `step_star_artist`, `trail_pre`, `trail_post`,
`timeline_cursor` in the artists list (filter out None).

### 4. STEP_VIS_ALL dispatch in render_one_trial()

Add module-level constant:

    STEP_VIS_TEMPLATES = ("phase_trail", "bos_phase", "star_only", "phase_bos")

In `render_one_trial()`, after resolving `step_vis = config.step_vis`:

    def _gif_path_for(suffix_sv: str, mode: str) -> Path:
        return gif_base.parent / f"{gif_base.name}__step_vis-{suffix_sv}__{mode}.gif"

    vis_list = list(STEP_VIS_TEMPLATES) if step_vis == "all" else [step_vis]

    for sv in vis_list:
        for mode in GIF_BOS_MODES:
            out_path = (
                gif_base.parent / f"{gif_base.name}__{safe_name(config.gif_name_suffix)}__{mode}.gif"
                if sv == "none"
                else _gif_path_for(sv, mode)
            )
            render_gif_animation(
                series=series,
                display=display,
                bos_polylines=bos_polylines,
                out_path=out_path,
                fps=config.fps,
                frame_step=config.frame_step,
                dpi=config.dpi,
                bos_mode=mode,
                x_lim=x_lim,
                y_lim=y_lim,
                trial_state_label=trial_state_label,
                step_vis=sv,
            )

When `step_vis == "none"` the existing naming is preserved (no `step_vis-` in filename).

### 5. Right panel text vertical position

In `apply_gif_right_panel()`, move text y from 0.97 to 0.92 and set
`verticalalignment="top"` to leave room for the timeline inset at the bottom.

## Concrete Steps

Run in the repo root (`c:\Users\Alice\OneDrive - 청주대학교\근전도 분석 코드\replace_V3D`):

    # 1. Syntax check after editing
    conda run -n module python -c "import ast; ast.parse(open('scripts/plot_bos_com_xy_sample.py').read()); print('OK')"

    # 2. Render all templates for default trial
    conda run -n module python scripts/plot_bos_com_xy_sample.py --step_vis all

    # Expected: 8 new GIF files under output/figures/bos_com_xy_sample/<subject>/
    #   *__step_vis-phase_trail__freeze.gif
    #   *__step_vis-phase_trail__live.gif
    #   *__step_vis-bos_phase__freeze.gif
    #   *__step_vis-bos_phase__live.gif
    #   *__step_vis-star_only__freeze.gif
    #   *__step_vis-star_only__live.gif
    #   *__step_vis-phase_bos__freeze.gif
    #   *__step_vis-phase_bos__live.gif

    # 3. Verify none mode is unchanged
    conda run -n module python scripts/plot_bos_com_xy_sample.py --step_vis none
    # Then MD5-compare against reference.

## Validation and Acceptance

- `--step_vis all` produces exactly 8 GIF files for the default trial (none pre-existing).
- Open `*__step_vis-phase_trail__freeze.gif`: gold star visible at step_onset position;
  COM trail turns from blue to orange at that point; timeline bar in right panel has
  orange vertical line; cursor moves across timeline as animation plays.
- Open `*__step_vis-bos_phase__freeze.gif`: BOS rectangle switches from blue to orange
  background at step_onset; gold star visible; timeline bar present.
- Open `*__step_vis-star_only__freeze.gif`: only gold star visible, trail and BOS unchanged.
- Open `*__step_vis-phase_bos__freeze.gif`: both trail split and BOS color change visible.
- `--step_vis none` output MD5 matches pre-change reference GIFs.
- `python -c "import ast; ast.parse(...)"` exits with code 0.

## Idempotence and Recovery

Re-running `--step_vis all` overwrites existing template GIFs (same filename, no duplicates).
`--step_vis none` always writes to the original naming, independent of template GIFs.
No database or irreversible state is modified.

## Artifacts and Notes

File to edit: `scripts/plot_bos_com_xy_sample.py`
New ExecPlan: `.codex/execplans/step_onset_vis_execplan.md`
Output GIFs: `output/figures/bos_com_xy_sample/<subject>/`

## Interfaces and Dependencies

No new pip dependencies. Uses existing matplotlib `inset_axes`, `FuncAnimation`, `PillowWriter`.
All imports already present in `scripts/plot_bos_com_xy_sample.py`.

New function signature added to the script:

    def add_timeline_inset(
        ax_side: plt.Axes,
        series: TrialSeries,
    ) -> matplotlib.lines.Line2D: ...

Modified function signature:

    def render_gif_animation(
        ...,
        step_vis: str = "none",
    ) -> int: ...

New `RenderConfig` field:

    step_vis: str = "none"
