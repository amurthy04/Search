"""Search, explained: a narrated Manim video. One Scene per chapter; rendered separately and concatenated.
AUDIO_DIR env var selects the narration (audio_edge or audio_cb); durations.json in it drives the timing."""
import json, os, math
import numpy as np
from manim import *

AUDIO = os.environ.get("AUDIO_DIR", "audio_edge")
DUR = json.load(open(os.path.join(AUDIO, "durations.json")))

config.background_color = "#0e1218"
BG = "#0e1218"
INK = "#e8ecf1"
MUTE = "#8a95a3"
LINE = "#2a3340"
BLUE = "#58a6ff"
TEAL = "#3ddbc0"
YEL = "#ffd166"
RED = "#ff6b6b"
VIOLET = "#b38cff"
F_SANS = "Avenir Next"
F_MONO = "Menlo"


def T(s, size=36, color=INK, weight="NORMAL", font=F_SANS, **kw):
    return Text(s, font=font, font_size=size, color=color, weight=weight, **kw)


def M(s, size=28, color=MUTE, **kw):
    return Text(s, font=F_MONO, font_size=size, color=color, **kw)


class Base(Scene):
    def setup(self):
        self.clock = 0.0
        self.seg_start = 0.0
        self.seg_dur = 1.0

    def narrate(self, key):
        self.add_sound(os.path.join(AUDIO, f"{key}.wav"))
        self.seg_start = self.clock
        self.seg_dur = DUR[key]

    def t(self, frac):
        """run_time as a fraction of the current narration segment"""
        return max(0.15, frac * self.seg_dur)

    def go(self, *anims, run_time=1.0, **kw):
        self.play(*anims, run_time=run_time, **kw)
        self.clock += run_time

    def hold(self, t):
        if t > 0.01:
            self.wait(t)
            self.clock += t

    def finish(self, gap=0.45):
        self.hold(max(0.1, self.seg_start + self.seg_dur - self.clock) + gap)

    def title(self, num, name):
        g = VGroup(M(f"{num}", 22, BLUE), T(name, 40, INK, weight="MEDIUM")).arrange(RIGHT, buff=0.3).to_corner(UL, buff=0.5)
        return g


def pipeline(scale=1.0, active=None):
    names = ["crawl", "passages", "embed", "index", "query"]
    cols = [BLUE, BLUE, TEAL, TEAL, YEL]
    boxes = VGroup()
    for n, c in zip(names, cols):
        r = RoundedRectangle(corner_radius=0.18, width=2.1, height=0.95).set_stroke(c, 2).set_fill(BG, 1)
        lab = T(n, 26, INK)
        boxes.add(VGroup(r, lab))
    boxes.arrange(RIGHT, buff=0.55)
    arrows = VGroup(*[Arrow(boxes[i][0].get_right(), boxes[i + 1][0].get_left(), buff=0.06, stroke_width=2.5, color=LINE,
                            max_tip_length_to_length_ratio=0.3) for i in range(len(names) - 1)])
    g = VGroup(boxes, arrows).scale(scale)
    if active is not None:
        for i, b in enumerate(boxes):
            if i != active:
                b.set_opacity(0.35)
    return g


# ============================================================ 1 INTRO
class C1Intro(Base):
    def construct(self):
        self.narrate("n1")
        q = T('"how do mRNA vaccines train the immune system?"', 34, YEL, slant=ITALIC).to_edge(UP, buff=1.0)
        self.go(Write(q), run_time=self.t(0.18))
        # a wall of pages
        rows, cols = 14, 44
        pages = VGroup(*[RoundedRectangle(corner_radius=0.03, width=0.22, height=0.3).set_stroke(width=0).set_fill(BLUE, 0.55)
                         for _ in range(rows * cols)]).arrange_in_grid(rows=rows, cols=cols, buff=0.06).scale(0.9).shift(DOWN * 0.6)
        self.go(LaggedStart(*[FadeIn(p, scale=0.6) for p in pages], lag_ratio=0.0015), run_time=self.t(0.3))
        lab = M("689,440 pages  ·  2,411,705 passages", 26, MUTE).next_to(pages, DOWN, buff=0.35)
        self.go(FadeIn(lab), run_time=self.t(0.08))
        # answer flashes: a handful of pages light up
        rng = np.random.default_rng(3)
        picks = rng.choice(len(pages), 10, replace=False)
        self.go(*[pages[i].animate.set_fill(YEL, 1).scale(1.6) for i in picks], run_time=self.t(0.12))
        timer = T("1.94 ms", 72, INK, weight="BOLD").move_to(pages.get_center())
        bgp = SurroundingRectangle(timer, color=BG, fill_color=BG, fill_opacity=0.85, buff=0.3).set_stroke(width=0)
        self.go(FadeIn(bgp), Write(timer), run_time=self.t(0.15))
        self.finish()

        self.narrate("n2")
        self.go(FadeOut(VGroup(q, pages, lab, timer, bgp)), run_time=self.t(0.1))
        pipe = pipeline(1.0).shift(UP * 0.4)
        boxes, arrows = pipe
        anims = []
        for i in range(5):
            anims.append(FadeIn(boxes[i], shift=UP * 0.3))
            if i < 4:
                anims.append(GrowArrow(arrows[i]))
        self.go(LaggedStart(*anims, lag_ratio=0.25), run_time=self.t(0.45))
        subs = ["321 pages/s", "×3.5 per page", "600 passages/s", "0.13 ms", "1.94 ms"]
        subl = VGroup(*[M(s, 20, MUTE).next_to(boxes[i], DOWN, buff=0.25) for i, s in enumerate(subs)])
        self.go(LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in subl], lag_ratio=0.2), run_time=self.t(0.2))
        self.finish(0.8)
        self.go(FadeOut(pipe), FadeOut(subl), run_time=0.6)


# ============================================================ 2 CRAWL
class C2Crawl(Base):
    def construct(self):
        ttl = self.title("01", "Crawl")
        self.add(ttl)
        # ---- n3: a request is mostly waiting
        self.narrate("n3")
        parts = [("dns", 0.5, VIOLET), ("connect", 0.9, BLUE), ("wait for first byte", 7.4, YEL), ("download", 0.5, TEAL)]
        bar = VGroup()
        x = -4.65
        for name, w, c in parts:
            r = Rectangle(width=w, height=0.7).set_stroke(width=0).set_fill(c, 0.9).move_to([x + w / 2, 0.6, 0])
            bar.add(r)
            x += w + 0.05
        labels = VGroup()
        for (name, w, c), r in zip(parts, bar):
            lab = T(name, 22, c).next_to(r, DOWN, buff=0.2)
            if w < 1.0:
                lab.rotate(-PI / 5, about_point=lab.get_top()).shift(DOWN * 0.1)
            labels.add(lab)
        self.go(LaggedStart(*[GrowFromEdge(r, LEFT) for r in bar], lag_ratio=0.9), run_time=self.t(0.5))
        self.go(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2), run_time=self.t(0.15))
        brace = Brace(bar, UP, color=MUTE)
        bl = M("~1 s median, 3.5 s at p90", 24, MUTE).next_to(brace, UP, buff=0.15)
        self.go(GrowFromCenter(brace), FadeIn(bl), run_time=self.t(0.12))
        self.finish()
        self.go(FadeOut(VGroup(bar, labels, brace, bl)), run_time=0.5)

        # ---- n4: event loop
        self.narrate("n4")
        loop = Circle(radius=1.5, color=LINE, stroke_width=3)
        ll = T("event loop", 24, MUTE).move_to(loop)
        self.go(Create(loop), FadeIn(ll), run_time=self.t(0.12))
        n = 48
        reqs = VGroup(*[Dot(loop.point_at_angle(i * TAU / n), radius=0.06, color=BLUE) for i in range(n)])
        rng = np.random.default_rng(1)
        for d in reqs:
            d.set_color([BLUE, YEL, YEL, YEL, TEAL][rng.integers(5)])
        hosts = VGroup(*[Dot(3.1 * np.array([math.cos(a), math.sin(a), 0]), radius=0.05, color=MUTE)
                         for a in np.linspace(0, TAU, 60, endpoint=False) + 0.05])
        self.go(LaggedStart(*[FadeIn(d, scale=0.3) for d in reqs], lag_ratio=0.02), FadeIn(hosts), run_time=self.t(0.25))
        reqs.add_updater(lambda m, dt: m.rotate(0.9 * dt, about_point=ORIGIN))
        # flashes: a request completes, a line to a host
        flashes = []
        for k in range(6):
            i = rng.integers(60)
            flashes.append(ShowPassingFlash(Line(ORIGIN, hosts[i].get_center(), color=TEAL, stroke_width=3), time_width=0.6))
        self.go(LaggedStart(*flashes, lag_ratio=0.3), run_time=self.t(0.3))
        note = M("hundreds in flight · one core", 24, MUTE).to_edge(DOWN, buff=0.7)
        self.go(FadeIn(note), run_time=self.t(0.1))
        self.finish()
        reqs.clear_updaters()

        # ---- n5: the new-connection ceiling
        self.narrate("n5")
        self.go(FadeOut(note), VGroup(loop, ll, reqs, hosts).animate.scale(0.55).to_edge(LEFT, buff=0.9), run_time=self.t(0.08))
        # new hosts popping up on the right
        rng2 = np.random.default_rng(5)
        new_hosts = VGroup(*[Dot([1.0 + rng2.random() * 5.0, -0.2 + rng2.random() * 2.6, 0], radius=0.055, color=MUTE) for _ in range(70)])
        lines = VGroup(*[Line(loop.get_center(), h.get_center(), color=LINE, stroke_width=1) for h in new_hosts])
        self.go(LaggedStart(*[AnimationGroup(FadeIn(h, scale=0.3), Create(l)) for h, l in zip(new_hosts, lines)], lag_ratio=0.03), run_time=self.t(0.22))
        # gauge
        gx0, gx1, gy = 1.0, 6.2, -1.6
        rail = Line([gx0, gy, 0], [gx1, gy, 0], color=LINE, stroke_width=6)
        limit = DashedLine([gx0 + (gx1 - gx0) * 0.6, gy - 0.35, 0], [gx0 + (gx1 - gx0) * 0.6, gy + 0.35, 0], color=RED, stroke_width=3)
        lim_l = M("120 / s", 20, RED).next_to(limit, UP, buff=0.1)
        gl = T("new connections per second", 20, MUTE).next_to(rail, DOWN, buff=0.25)
        vt = ValueTracker(0.0)
        needle = always_redraw(lambda: Triangle(color=YEL, fill_opacity=1).scale(0.14).rotate(PI).move_to([gx0 + (gx1 - gx0) * vt.get_value(), gy + 0.28, 0]))
        self.go(Create(rail), Create(limit), FadeIn(lim_l), FadeIn(gl), FadeIn(needle), run_time=self.t(0.08))
        self.go(vt.animate.set_value(0.95), run_time=self.t(0.2), rate_func=rate_functions.ease_in_out_sine)
        # SYN queue piles up: lines turn red, stack of boxes
        self.go(*[l.animate.set_color(RED).set_stroke(width=1.5) for l in lines[::2]], run_time=self.t(0.08))
        q = VGroup(*[Square(0.18).set_fill(RED, 0.9).set_stroke(width=0) for _ in range(24)]).arrange_in_grid(rows=3, cols=8, buff=0.06).next_to(gl, DOWN, buff=0.3)
        ql = M("SYN queue · connect 0.15 s → 7 s", 20, RED).next_to(q, DOWN, buff=0.15)
        self.go(LaggedStart(*[FadeIn(s, shift=DOWN * 0.1) for s in q], lag_ratio=0.05), FadeIn(ql), run_time=self.t(0.2))
        self.finish()

        # ---- n6: warm first, cold metered
        self.narrate("n6")
        self.go(FadeOut(VGroup(new_hosts, lines, rail, limit, lim_l, gl, needle, q, ql)), run_time=self.t(0.08))
        warm_box = RoundedRectangle(corner_radius=0.2, width=2.6, height=1.9).set_stroke(TEAL, 2).move_to([2.2, 1.55, 0])
        cold_box = RoundedRectangle(corner_radius=0.2, width=2.6, height=1.9).set_stroke(MUTE, 2).move_to([2.2, -1.75, 0])
        wl = T("warm hosts", 24, TEAL).next_to(warm_box, UP, buff=0.12)
        cl = T("cold hosts", 24, MUTE).next_to(cold_box, UP, buff=0.12)
        wsub = M("keep-alive · no handshake", 18, MUTE).next_to(warm_box, DOWN, buff=0.1)
        csub = M("DNS + TCP + TLS", 18, MUTE).next_to(cold_box, DOWN, buff=0.1)
        rng3 = np.random.default_rng(9)
        wdots = VGroup(*[Dot(warm_box.get_center() + [(rng3.random() - 0.5) * 2.1, (rng3.random() - 0.5) * 1.7, 0], radius=0.06, color=TEAL) for _ in range(18)])
        cdots = VGroup(*[Dot(cold_box.get_center() + [(rng3.random() - 0.5) * 2.1, (rng3.random() - 0.5) * 1.7, 0], radius=0.06, color=MUTE) for _ in range(40)])
        self.go(Create(warm_box), Create(cold_box), FadeIn(wl), FadeIn(cl), FadeIn(wsub), FadeIn(csub), FadeIn(wdots), FadeIn(cdots), run_time=self.t(0.15))
        # arrows into the loop: warm direct, cold through bucket
        bucket = RoundedRectangle(corner_radius=0.1, width=0.9, height=1.3).set_stroke(YEL, 2).move_to([-0.3, -1.6, 0])
        fill = Rectangle(width=0.86, height=0.5).set_fill(YEL, 0.8).set_stroke(width=0).move_to(bucket.get_bottom() + UP * 0.27)
        bl2 = M("token bucket", 18, YEL).next_to(bucket, DOWN, buff=0.1)
        a_w = Arrow(warm_box.get_left(), loop.get_right() + UP * 0.3, buff=0.1, color=TEAL, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        a_c1 = Arrow(cold_box.get_left(), bucket.get_right(), buff=0.1, color=MUTE, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        a_c2 = Arrow(bucket.get_left(), loop.get_right() + DOWN * 0.3, buff=0.1, color=YEL, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        self.go(GrowArrow(a_w), run_time=self.t(0.08))
        first = M("first", 18, TEAL).next_to(a_w, UP, buff=0.05)
        self.go(FadeIn(first), run_time=self.t(0.05))
        self.go(Create(bucket), FadeIn(fill), FadeIn(bl2), GrowArrow(a_c1), GrowArrow(a_c2), run_time=self.t(0.12))
        # tokens drip: dots travel cold -> bucket -> loop at a steady rate
        drips = []
        drip_dots = VGroup()
        for k in range(5):
            d = cdots[k].copy().set_color(YEL)
            drip_dots.add(d)
            path = VMobject().set_points_as_corners([d.get_center(), bucket.get_center(), loop.get_right() + DOWN * 0.3])
            drips.append(MoveAlongPath(d, path))
        self.go(LaggedStart(*drips, lag_ratio=0.35), run_time=self.t(0.3))
        self.finish()

        # ---- n7: 52 -> 321
        self.narrate("n7")
        keep = VGroup(loop, ll, reqs, hosts, warm_box, cold_box, wl, cl, wsub, csub, wdots, cdots, bucket, fill, bl2, a_w, a_c1, a_c2, first, drip_dots)
        self.go(FadeOut(keep), run_time=self.t(0.08))
        base_y = -1.8
        ax = Line([-3.5, base_y, 0], [3.5, base_y, 0], color=LINE)
        b1 = Rectangle(width=1.6, height=0.01).set_fill(MUTE, 0.9).set_stroke(width=0).move_to([-1.6, base_y, 0], aligned_edge=DOWN)
        b2 = Rectangle(width=1.6, height=0.01).set_fill(TEAL, 0.9).set_stroke(width=0).move_to([1.6, base_y, 0], aligned_edge=DOWN)
        l1 = T("without", 24, MUTE).next_to(b1, DOWN, buff=0.2)
        l2 = T("with the bucket", 24, TEAL).next_to(b2, DOWN, buff=0.2)
        self.go(Create(ax), FadeIn(l1), FadeIn(l2), run_time=self.t(0.1))
        v1, v2 = ValueTracker(0), ValueTracker(0)
        n1 = always_redraw(lambda: M(f"{v1.get_value():.0f} pages/s", 26, INK).next_to(b1, UP, buff=0.15))
        n2 = always_redraw(lambda: M(f"{v2.get_value():.0f} pages/s", 26, INK).next_to(b2, UP, buff=0.15))
        self.add(n1, n2)
        self.go(b1.animate.stretch_to_fit_height(52 / 321 * 3.6).move_to([-1.6, base_y, 0], aligned_edge=DOWN), v1.animate.set_value(52), run_time=self.t(0.2))
        self.go(b2.animate.stretch_to_fit_height(3.6).move_to([1.6, base_y, 0], aligned_edge=DOWN), v2.animate.set_value(321), run_time=self.t(0.3), rate_func=rate_functions.ease_out_cubic)
        self.finish(0.8)
        self.go(FadeOut(VGroup(ax, b1, b2, l1, l2, n1, n2, ttl)), run_time=0.6)


# ============================================================ 3 PASSAGES
class C3Passages(Base):
    def construct(self):
        ttl = self.title("02", "Passages")
        self.add(ttl)
        self.narrate("n8")
        rng = np.random.default_rng(11)
        page = RoundedRectangle(corner_radius=0.15, width=4.2, height=5.6).set_stroke(LINE, 2).move_to([-3.3, -0.2, 0])
        title_line = Rectangle(width=2.6, height=0.16).set_fill(YEL, 1).set_stroke(width=0).move_to(page.get_top() + DOWN * 0.45 + LEFT * 0.6)
        lines = VGroup()
        for i in range(22):
            w = 3.3 if rng.random() > 0.2 else 1.6 + rng.random() * 1.6
            ln = Rectangle(width=w, height=0.1).set_fill(MUTE, 0.7).set_stroke(width=0)
            ln.move_to(page.get_top() + DOWN * (0.95 + i * 0.215) + LEFT * (3.3 - w) / 2 + LEFT * 0.1)
            lines.add(ln)
        self.go(Create(page), FadeIn(title_line), LaggedStart(*[FadeIn(l) for l in lines], lag_ratio=0.03), run_time=self.t(0.18))
        tl = M("title", 18, YEL).next_to(title_line, RIGHT, buff=0.15)
        self.add(tl)
        # window of 7 lines, step 5 (overlap 2)
        win_h = 7 * 0.215
        win = Rectangle(width=3.8, height=win_h + 0.05).set_fill(BLUE, 0.18).set_stroke(BLUE, 2)
        win.move_to(lines[0].get_top() + DOWN * win_h / 2 + RIGHT * 0.0).set_x(page.get_x())
        self.go(FadeIn(win), run_time=self.t(0.08))
        wlab = M("180 words", 18, BLUE).next_to(win, RIGHT, buff=0.15)
        self.add(wlab)
        col_x = 2.4
        passages = VGroup()
        step = 5
        for k in range(4):
            top_i = k * step
            if k > 0:
                self.go(win.animate.shift(DOWN * step * 0.215), wlab.animate.shift(DOWN * step * 0.215), run_time=self.t(0.06))
            chunk = VGroup(*[lines[i].copy() for i in range(top_i, min(22, top_i + 7))])
            card = RoundedRectangle(corner_radius=0.08, width=2.9, height=1.25).set_stroke(BLUE, 1.5).set_fill(BG, 1)
            card.move_to([col_x, 2.1 - k * 1.45, 0])
            tag = title_line.copy().stretch_to_fit_width(0.9).stretch_to_fit_height(0.12).move_to(card.get_top() + DOWN * 0.2 + LEFT * 0.9)
            mini = chunk.copy().scale(0.55).arrange(DOWN, buff=0.06, aligned_edge=LEFT).next_to(tag, DOWN, buff=0.08, aligned_edge=LEFT)
            self.go(FadeIn(card), TransformFromCopy(chunk, mini), TransformFromCopy(title_line, tag), run_time=self.t(0.08))
            self.add(tag, mini)
            passages.add(VGroup(card, tag, mini))
            if k == 1:
                ov = M("overlap", 16, TEAL).next_to(win, LEFT, buff=0.1).shift(UP * (win_h / 2 - 0.2))
                ovr = Rectangle(width=3.8, height=2 * 0.215).set_fill(TEAL, 0.25).set_stroke(width=0).move_to(win.get_top() + DOWN * 0.215)
                self.go(FadeIn(ovr), FadeIn(ov), run_time=self.t(0.05))
                self.go(FadeOut(ovr), FadeOut(ov), run_time=self.t(0.03))
        more = M("… up to 6 per page", 20, MUTE).next_to(passages, DOWN, buff=0.25)
        tagl = M("title + chunk → one passage", 20, YEL).next_to(passages, UP, buff=0.25)
        self.go(FadeIn(more), FadeIn(tagl), run_time=self.t(0.08))
        self.finish(0.8)
        self.go(FadeOut(Group(*self.mobjects)), run_time=0.6)


# ============================================================ 4 EMBED
class C4Embed(Base):
    def construct(self):
        ttl = self.title("03", "Embed")
        self.add(ttl)
        # ---- n9: passage -> transformer -> vector -> unit length
        self.narrate("n9")
        toks = ["[CLS]", "how", "do", "m", "##rna", "vaccines", "train", "the", "immune", "system"]
        tg = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.08, width=0.9 + 0.05 * len(t), height=0.5).set_stroke(BLUE, 1.5),
                              M(t, 18, INK)) for t in toks]).arrange(RIGHT, buff=0.1).scale(0.75).to_edge(UP, buff=1.4)
        self.go(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tg], lag_ratio=0.08), run_time=self.t(0.15))
        layers = VGroup(*[RoundedRectangle(corner_radius=0.06, width=5.5, height=0.22).set_stroke(width=0).set_fill(LINE, 1) for _ in range(12)]).arrange(DOWN, buff=0.07).move_to([-2.2, -0.9, 0])
        ll = M("12 transformer layers · 33M parameters", 18, MUTE).next_to(layers, DOWN, buff=0.15)
        a1 = Arrow(tg.get_bottom(), layers.get_top(), buff=0.1, color=MUTE, stroke_width=2, max_tip_length_to_length_ratio=0.25)
        self.go(GrowArrow(a1), FadeIn(layers), FadeIn(ll), run_time=self.t(0.1))
        self.go(LaggedStart(*[Succession(l.animate.set_fill(BLUE, 1), l.animate.set_fill("#1f3b63", 1)) for l in layers], lag_ratio=0.15), run_time=self.t(0.22))
        # vector bars
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 48)
        vx = 2.4
        bars = VGroup()
        for i, v in enumerate(vals):
            b = Rectangle(width=0.07, height=abs(v) * 0.42 + 0.02).set_stroke(width=0).set_fill(TEAL if v > 0 else BLUE, 0.95)
            b.move_to([vx - 1.7 + i * 0.075, -0.9 + (abs(v) * 0.42 + 0.02) / 2 * (1 if v > 0 else -1), 0])
            bars.add(b)
        axis = Line([vx - 1.8, -0.9, 0], [vx + 1.9, -0.9, 0], color=LINE)
        vl = M("384 numbers", 18, MUTE).next_to(axis, DOWN, buff=0.95)
        a2 = Arrow(layers.get_right(), axis.get_left(), buff=0.15, color=MUTE, stroke_width=2, max_tip_length_to_length_ratio=0.25)
        self.go(GrowArrow(a2), Create(axis), LaggedStart(*[GrowFromEdge(b, DOWN if vals[i] > 0 else UP) for i, b in enumerate(bars)], lag_ratio=0.02), FadeIn(vl), run_time=self.t(0.2))
        # normalize: shrink bars uniformly
        s = 1 / np.sqrt((vals ** 2).sum()) * 6.0
        norm_l = M("÷ length  →  |v| = 1", 20, YEL).next_to(vl, DOWN, buff=0.1)
        self.go(*[b.animate.stretch_to_fit_height(max(0.02, abs(vals[i]) * 0.42 * s)).move_to([vx - 1.7 + i * 0.075, -0.9 + max(0.02, abs(vals[i]) * 0.42 * s) / 2 * (1 if vals[i] > 0 else -1), 0]) for i, b in enumerate(bars)],
                FadeIn(norm_l), run_time=self.t(0.12))
        self.finish()

        # ---- n10: points on a sphere
        self.narrate("n10")
        self.go(FadeOut(VGroup(tg, layers, ll, a1, a2, bars, axis, vl, norm_l)), run_time=self.t(0.08))
        circ = Circle(radius=2.6, color=LINE, stroke_width=2).move_to([-2.0, -0.3, 0])
        self.go(Create(circ), run_time=self.t(0.12))
        rng = np.random.default_rng(8)
        angs = np.concatenate([rng.normal(0.9, 0.18, 14), rng.normal(2.6, 0.2, 12), rng.normal(4.4, 0.25, 12), rng.uniform(0, TAU, 22)])
        pts = VGroup(*[Dot(circ.point_at_angle(a % TAU), radius=0.055, color=TEAL) for a in angs])
        self.go(LaggedStart(*[FadeIn(p, scale=0.3) for p in pts], lag_ratio=0.02), run_time=self.t(0.2))
        cl1 = M("medicine", 18, MUTE).move_to(circ.point_at_angle(1.1) + (circ.point_at_angle(1.1) - circ.get_center()) * 0.3)
        cl2 = M("programming", 18, MUTE).move_to(circ.point_at_angle(2.6) + (circ.point_at_angle(2.6) - circ.get_center()) * 0.28)
        cl3 = M("history", 18, MUTE).move_to(circ.point_at_angle(4.4) + (circ.point_at_angle(4.4) - circ.get_center()) * 0.22)
        self.go(FadeIn(cl1), FadeIn(cl2), FadeIn(cl3), run_time=self.t(0.1))
        qa = 0.78
        qarrow = Arrow(circ.get_center(), circ.point_at_angle(qa), buff=0, color=YEL, stroke_width=5, max_tip_length_to_length_ratio=0.1)
        ql = T("question", 22, YEL).next_to(qarrow.get_end(), UR, buff=0.1).shift(UP * 0.25)
        self.go(GrowArrow(qarrow), FadeIn(ql), run_time=self.t(0.12))
        best = min(pts, key=lambda p: np.linalg.norm(p.get_center() - circ.point_at_angle(qa)))
        parrow = Arrow(circ.get_center(), best.get_center(), buff=0, color=TEAL, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        pl = T("answer passage", 22, TEAL).next_to(parrow.get_end(), RIGHT, buff=0.15)
        self.go(GrowArrow(parrow), FadeIn(pl), Flash(best, color=TEAL), run_time=self.t(0.15))
        self.finish()

        # ---- n11: cosine
        self.narrate("n11")
        self.go(FadeOut(VGroup(pts, cl1, cl2, cl3, ql, pl)), run_time=self.t(0.08))
        theta = ValueTracker(0.35)
        base_a = 0.5
        v1 = always_redraw(lambda: Arrow(circ.get_center(), circ.point_at_angle(base_a), buff=0, color=TEAL, stroke_width=4, max_tip_length_to_length_ratio=0.1))
        v2 = always_redraw(lambda: Arrow(circ.get_center(), circ.point_at_angle(base_a + theta.get_value()), buff=0, color=YEL, stroke_width=5, max_tip_length_to_length_ratio=0.1))
        arc = always_redraw(lambda: Arc(radius=0.9, start_angle=base_a, angle=theta.get_value(), arc_center=circ.get_center(), color=BLUE, stroke_width=3))
        thl = always_redraw(lambda: M("θ", 24, BLUE).move_to(circ.get_center() + 1.2 * np.array([math.cos(base_a + theta.get_value() / 2), math.sin(base_a + theta.get_value() / 2), 0])))
        self.remove(qarrow, parrow)
        self.add(v1, v2, arc, thl)
        eq_top = M("q · p  =  cos θ", 30, INK).move_to([3.3, 0.6, 0])
        eq_val = always_redraw(lambda: M(f"=  {math.cos(theta.get_value()):.3f}", 36, YEL).next_to(eq_top, DOWN, buff=0.35, aligned_edge=LEFT))
        eq = VGroup(eq_top, eq_val)
        self.go(FadeIn(eq), run_time=self.t(0.1))
        self.go(theta.animate.set_value(0.02), run_time=self.t(0.22), rate_func=rate_functions.ease_in_out_sine)
        self.go(theta.animate.set_value(PI / 2), run_time=self.t(0.3), rate_func=rate_functions.ease_in_out_sine)
        self.go(theta.animate.set_value(0.5), run_time=self.t(0.15))
        self.finish()

        # ---- n12: number line of real scores
        self.narrate("n12")
        self.go(FadeOut(VGroup(circ, v1, v2, arc, thl, eq)), run_time=self.t(0.08))
        nl = NumberLine(x_range=[0.6, 1.0, 0.1], length=10, color=MUTE, include_numbers=False, include_tip=False).shift(DOWN * 0.3)
        ticks = VGroup(*[M(f"{v:.1f}", 20, MUTE).next_to(nl.n2p(v), DOWN, buff=0.2) for v in [0.6, 0.7, 0.8, 0.9, 1.0]])
        self.go(Create(nl), FadeIn(ticks), run_time=self.t(0.1))
        def mark(v, label, color, up=True, size=20):
            d = Dot(nl.n2p(v), radius=0.09, color=color)
            l = T(label, size, color).next_to(d, UP if up else DOWN, buff=0.25)
            return VGroup(d, l)
        m1 = mark(0.875, "mRNA → Scientific American   0.875", TEAL, True)
        m2 = mark(0.816, "→ MedlinePlus  0.816", TEAL, False).shift(DOWN * 0.5)
        self.go(FadeIn(m1, shift=DOWN * 0.2), FadeIn(m2, shift=UP * 0.2), run_time=self.t(0.2))
        m3 = mark(0.742, "Westphalia (not crawled)  0.742", RED, True)
        m3[1].shift(UP * 0.55)
        self.go(FadeIn(m3, shift=DOWN * 0.2), run_time=self.t(0.15))
        band_ok = Rectangle(width=nl.n2p(1.0)[0] - nl.n2p(0.8)[0], height=0.5).set_fill(TEAL, 0.15).set_stroke(width=0).move_to(nl.n2p(0.9))
        band_bad = Rectangle(width=nl.n2p(0.75)[0] - nl.n2p(0.6)[0], height=0.5).set_fill(RED, 0.15).set_stroke(width=0).move_to(nl.n2p(0.675))
        ok_l = M("every correct hit", 20, TEAL).next_to(band_ok, DOWN, buff=1.1)
        bad_l = M("every miss", 20, RED).next_to(band_bad, DOWN, buff=1.1)
        self.go(FadeIn(band_ok), FadeIn(ok_l), run_time=self.t(0.1))
        self.go(FadeIn(band_bad), FadeIn(bad_l), run_time=self.t(0.1))
        self.finish()

        # ---- n13: throughput
        self.narrate("n13")
        self.go(FadeOut(VGroup(nl, ticks, m1, m2, m3, band_ok, band_bad, ok_l, bad_l)), run_time=self.t(0.08))
        data = [("MLX", 159, MUTE), ("torch on Metal, fp16", 690, BLUE), ("hand-written BERT → CoreML GPU", 1491, TEAL)]
        rows = VGroup()
        for name, v, c in data:
            lab = T(name, 24, INK)
            bar = Rectangle(width=0.01, height=0.5).set_fill(c, 0.9).set_stroke(width=0)
            num = M(f"{v:,} passages/s", 22, c)
            rows.add(VGroup(lab, bar, num))
        y = 1.3
        bars_anim = []
        for (name, v, c), r in zip(data, rows):
            lab, bar, num = r
            lab.move_to([-5.5, y, 0], aligned_edge=LEFT)
            bar.move_to([-5.5, y - 0.55, 0], aligned_edge=LEFT)
            num.move_to([-5.5 + v / 1491 * 8.5 + 0.2, y - 0.55, 0], aligned_edge=LEFT)
            bars_anim.append(AnimationGroup(FadeIn(lab), bar.animate.stretch_to_fit_width(v / 1491 * 8.5).move_to([-5.5, y - 0.55, 0], aligned_edge=LEFT), FadeIn(num)))
            y -= 1.5
        self.go(LaggedStart(*bars_anim, lag_ratio=0.5), run_time=self.t(0.45))
        note = M("2.41M passages · same weights, same accuracy · no training", 20, MUTE).to_edge(DOWN, buff=0.7)
        self.go(FadeIn(note), run_time=self.t(0.1))
        self.finish(0.8)
        self.go(FadeOut(Group(*self.mobjects)), run_time=0.6)


# ============================================================ 5 HNSW
def knn_graph(P, k):
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(D, axis=1)[:, :k]
    return nb


def greedy(P, nb, start, q):
    cur = start
    path = [cur]
    while True:
        d = ((P[cur] - q) ** 2).sum()
        best = min(nb[cur], key=lambda j: ((P[j] - q) ** 2).sum())
        if ((P[best] - q) ** 2).sum() < d:
            cur = best
            path.append(cur)
        else:
            return path


class C5Index(Base):
    def construct(self):
        ttl = self.title("04", "Index")
        self.add(ttl)
        rng = np.random.default_rng(2024)
        # ---- n14: brute force
        self.narrate("n14")
        P = np.column_stack([rng.uniform(-6.0, 6.0, 230), rng.uniform(-2.6, 2.2, 230)])
        dots = VGroup(*[Dot([x, y, 0], radius=0.05, color=TEAL) for x, y in P])
        q = np.array([2.1, 0.6])
        qm = Star(n=4, outer_radius=0.22, inner_radius=0.1, color=YEL, fill_opacity=1).move_to([*q, 0])
        self.go(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.004), FadeIn(qm), run_time=self.t(0.2))
        scan = ValueTracker(0)
        order = np.argsort(P[:, 0])
        def upd(m):
            k = int(scan.get_value())
            for idx, i in enumerate(order):
                m[i].set_color(MUTE if idx < k else TEAL)
        dots.add_updater(upd)
        cnt = always_redraw(lambda: M(f"{int(scan.get_value() / 230 * 2411705):,} comparisons", 24, MUTE).to_edge(DOWN, buff=0.7))
        self.add(cnt)
        self.go(scan.animate.set_value(230), run_time=self.t(0.4), rate_func=linear)
        dots.clear_updaters()
        top = np.argsort(((P - q) ** 2).sum(1))[:10]
        self.go(*[dots[i].animate.set_color(YEL).scale(1.6) for i in top], run_time=self.t(0.1))
        tm = T("16 ms", 40, INK, weight="BOLD").to_corner(UR, buff=0.6)
        self.go(FadeIn(tm), run_time=self.t(0.08))
        self.finish()

        # ---- n15: greedy walk on a graph
        self.narrate("n15")
        self.go(FadeOut(cnt), FadeOut(tm), *[dots[i].animate.set_color(TEAL).scale(1 / 1.6) for i in top], run_time=self.t(0.06))
        keep = rng.choice(230, 90, replace=False)
        P2 = P[keep]
        nb = knn_graph(P2, 4)
        edges = VGroup()
        for i in range(len(P2)):
            for j in nb[i]:
                if j > i or i not in nb[j]:
                    edges.add(Line([*P2[i], 0], [*P2[j], 0], color=LINE, stroke_width=1.2))
        self.go(FadeOut(VGroup(*[dots[i] for i in range(230) if i not in set(keep)])), run_time=self.t(0.08))
        self.go(Create(edges, lag_ratio=0.01), run_time=self.t(0.18))
        start = int(np.argmin(P2[:, 0]))
        path = greedy(P2, nb, start, q)
        cur = Circle(radius=0.18, color=YEL, stroke_width=3).move_to([*P2[start], 0])
        self.go(Create(cur), run_time=self.t(0.05))
        for a, b in zip(path[:-1], path[1:]):
            nbs = VGroup(*[Line([*P2[a], 0], [*P2[j], 0], color=BLUE, stroke_width=2.5) for j in nb[a]])
            self.go(FadeIn(nbs), run_time=self.t(0.03))
            step = Line([*P2[a], 0], [*P2[b], 0], color=YEL, stroke_width=3.5)
            self.go(FadeOut(nbs), Create(step), cur.animate.move_to([*P2[b], 0]), run_time=self.t(0.05))
        self.go(Flash(cur, color=YEL, line_length=0.3), run_time=self.t(0.08))
        stl = M(f"{len(path) - 1} steps · {len(path) * 4} distances", 24, MUTE).to_edge(DOWN, buff=0.7)
        self.go(FadeIn(stl), run_time=self.t(0.06))
        self.finish()

        # ---- n16: layers
        self.narrate("n16")
        self.go(FadeOut(Group(*[m for m in self.mobjects if m is not ttl])), run_time=self.t(0.06))
        def proj(x, y, layer):
            return np.array([x * 0.6 + y * 0.35 + 0.9, y * 0.3 + (layer - 1) * 2.0 - 0.2, 0])
        planes = VGroup(*[Polygon(proj(-6, -2.6, l), proj(6, -2.6, l), proj(6, 2.4, l), proj(-6, 2.4, l)).set_stroke(LINE, 1.5).set_fill(BLUE, 0.04) for l in range(3)])
        lvl = rng.random(len(P2))
        L = np.where(lvl < 0.08, 2, np.where(lvl < 0.35, 1, 0))
        layer_dots = []
        layer_edges = []
        for l in range(3):
            idx = np.where(L >= l)[0]
            sub = P2[idx]
            nbl = knn_graph(sub, 3 if l else 4)
            ld = VGroup(*[Dot(proj(*sub[i], l), radius=0.045 if l == 0 else 0.07, color=TEAL if l == 0 else BLUE) for i in range(len(sub))])
            le = VGroup()
            for i in range(len(sub)):
                for j in nbl[i]:
                    if j > i:
                        le.add(Line(proj(*sub[i], l), proj(*sub[j], l), color=LINE, stroke_width=1))
            layer_dots.append((idx, ld)); layer_edges.append(le)
        self.go(LaggedStart(*[AnimationGroup(Create(planes[l]), FadeIn(layer_edges[l]), FadeIn(layer_dots[l][1])) for l in [0, 1, 2]], lag_ratio=0.5), run_time=self.t(0.3))
        ll = VGroup(*[M(s, 18, MUTE).next_to(planes[l], LEFT, buff=0.2) for l, s in enumerate(["layer 0 · everything", "layer 1 · 1 in M", "layer 2 · a handful"])])
        self.go(FadeIn(ll), run_time=self.t(0.06))
        # descend
        qpts = [proj(*q, l) for l in range(3)]
        qstars = VGroup(*[Star(n=4, outer_radius=0.16, inner_radius=0.07, color=YEL, fill_opacity=0.9).move_to(p) for p in qpts])
        self.go(FadeIn(qstars), run_time=self.t(0.05))
        ep = None
        trail = VGroup()
        for l in [2, 1, 0]:
            idx, ld = layer_dots[l]
            sub = P2[idx]
            nbl = knn_graph(sub, 3 if l else 4)
            if ep is None:
                s_local = int(np.argmin(sub[:, 0]))
            else:
                s_local = int(np.where(idx == ep)[0][0])
            pth = greedy(sub, nbl, s_local, q)
            pl = VMobject(color=YEL, stroke_width=3.5).set_points_as_corners([proj(*sub[i], l) for i in pth] if len(pth) > 1 else [proj(*sub[pth[0]], l), proj(*sub[pth[0]], l) + 0.001])
            self.go(Create(pl), run_time=self.t(0.07 if l else 0.1))
            trail.add(pl)
            ep = int(idx[pth[-1]])
            if l > 0:
                drop = DashedLine(proj(*P2[ep], l), proj(*P2[ep], l - 1), color=YEL, stroke_width=2)
                self.go(Create(drop), run_time=self.t(0.04))
                trail.add(drop)
        self.go(Flash(proj(*P2[ep], 0), color=YEL, line_length=0.3), run_time=self.t(0.06))
        self.finish()

        # ---- n17: ef trade-off
        self.narrate("n17")
        self.go(FadeOut(Group(*[m for m in self.mobjects if m is not ttl])), run_time=self.t(0.06))
        ax = Axes(x_range=[0, 0.32, 0.1], y_range=[0.95, 1.0, 0.01], x_length=8.5, y_length=4.4, axis_config={"include_numbers": False, "include_tip": False, "color": MUTE}).shift(DOWN * 0.3 + LEFT * 0.3)
        xl = M("latency per query, ms", 20, MUTE).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = M("recall@10", 20, MUTE).next_to(ax.y_axis, LEFT, buff=0.1).rotate(PI / 2).shift(LEFT * 0.4)
        xt = VGroup(*[M(f"{v:.1f}", 18, MUTE).next_to(ax.c2p(v, 0.95), DOWN, buff=0.15) for v in [0.0, 0.1, 0.2, 0.3]])
        yt = VGroup(*[M(f"{v:.2f}", 18, MUTE).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in [0.96, 0.98, 1.0]])
        self.go(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), run_time=self.t(0.12))
        pts = [(16, 0.06, 0.965), (32, 0.09, 0.984), (64, 0.13, 0.991), (128, 0.27, 0.994)]
        ds = VGroup()
        for ef, ms, r in pts:
            d = Dot(ax.c2p(ms, r), radius=0.1, color=TEAL if ef == 64 else BLUE)
            l = M(f"ef {ef}", 18, TEAL if ef == 64 else MUTE).next_to(d, UP if ef != 128 else DOWN, buff=0.15)
            ds.add(VGroup(d, l))
        curve = VMobject(color=LINE, stroke_width=2).set_points_smoothly([ax.c2p(ms, r) for _, ms, r in pts])
        self.go(Create(curve), LaggedStart(*[FadeIn(d, scale=0.5) for d in ds], lag_ratio=0.3), run_time=self.t(0.3))
        box = SurroundingRectangle(ds[2], color=TEAL, buff=0.15, corner_radius=0.1)
        exact = M("exact: 16 ms, recall 1.0 →", 20, MUTE).to_corner(UR, buff=0.6).shift(DOWN * 0.6)
        pick = VGroup(T("ef = 64", 30, TEAL, weight="MEDIUM"), M("0.991 recall · 0.13 ms", 20, INK), M("120× faster than exact", 20, INK)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(ax.c2p(0.22, 0.962))
        self.go(Create(box), FadeIn(exact), run_time=self.t(0.1))
        self.go(FadeIn(pick), run_time=self.t(0.1))
        self.finish(0.8)
        self.go(FadeOut(Group(*self.mobjects)), run_time=0.6)


# ============================================================ 6 QUERY
class C6Query(Base):
    def construct(self):
        ttl = self.title("05", "Query")
        self.add(ttl)
        self.narrate("n18")
        q = T('"how do mRNA vaccines train the immune system?"', 30, YEL, slant=ITALIC).to_edge(UP, buff=1.3)
        self.go(Write(q), run_time=self.t(0.08))
        stages = [("embed the question", 1.68, BLUE), ("walk the graph", 0.19, TEAL), ("collapse duplicates", 0.03, VIOLET), ("read snippets (mmap)", 0.02, MUTE)]
        total = sum(s[1] for s in stages)
        scale = 9.5 / total
        x0 = -4.75
        y = 0.9
        clock = ValueTracker(0.0)
        cl = always_redraw(lambda: T(f"{clock.get_value():.2f} ms", 54, INK, weight="BOLD").to_edge(DOWN, buff=0.8))
        self.add(cl)
        acc = 0.0
        self.go(FadeIn(cl), run_time=self.t(0.02))
        drawn = VGroup()
        for name, ms, c in stages:
            lab = T(name, 24, c).move_to([x0, y + 0.45, 0], aligned_edge=LEFT)
            ghost = Rectangle(width=ms * scale, height=0.42).set_stroke(c, 1, opacity=0.5).set_fill(c, 0.08).move_to([x0 + acc * scale, y, 0], aligned_edge=LEFT)
            bar = Rectangle(width=0.01, height=0.42).set_stroke(width=0).set_fill(c, 0.95).move_to([x0 + acc * scale, y, 0], aligned_edge=LEFT)
            num = M(f"{ms:.2f} ms", 22, c)
            if ms * scale > 1.2:
                num.move_to(ghost.get_center())
            else:
                num.next_to(ghost, RIGHT, buff=0.15)
            self.go(FadeIn(lab), FadeIn(ghost), run_time=self.t(0.02))
            rt = self.t(0.55 * ms / total) if ms > 0.1 else self.t(0.04)
            self.go(bar.animate.stretch_to_fit_width(max(0.02, ms * scale)).move_to([x0 + acc * scale, y, 0], aligned_edge=LEFT), clock.animate.set_value(acc + ms), run_time=rt, rate_func=linear)
            self.go(FadeIn(num), run_time=self.t(0.02))
            acc += ms
            y -= 1.1
            drawn.add(lab, ghost, bar, num)
        self.finish()

        self.narrate("n19")
        tot = T("1.94 ms", 54, INK, weight="BOLD").to_edge(DOWN, buff=0.8)
        self.remove(cl)
        self.add(tot)
        self.go(FadeOut(drawn), FadeOut(q), run_time=self.t(0.1))
        ring = VGroup()
        start = PI / 2
        for name, ms, c in stages:
            ang = -TAU * ms / total
            sec = AnnularSector(inner_radius=1.3, outer_radius=2.2, angle=ang, start_angle=start, color=c, fill_opacity=0.9).set_stroke(BG, 2)
            ring.add(sec)
            start += ang
        ring.move_to([0, 0.3, 0])
        lab85 = VGroup(T("85%", 44, BLUE, weight="BOLD"), M("the encoder", 20, MUTE)).arrange(DOWN, buff=0.1).move_to(ring)
        side = VGroup(M("10%  graph walk", 22, TEAL), M("1.5%  dedupe", 22, VIOLET), M("1%  snippets", 22, MUTE)).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(ring, RIGHT, buff=0.8)
        self.go(LaggedStart(*[FadeIn(s) for s in ring], lag_ratio=0.2), run_time=self.t(0.3))
        self.go(FadeIn(lab85), FadeIn(side), run_time=self.t(0.2))
        self.finish(0.8)
        self.go(FadeOut(Group(*self.mobjects)), run_time=0.6)


# ============================================================ 7 OUTRO
class C7Outro(Base):
    def construct(self):
        self.narrate("n20")
        rows = [("crawl", "46", "321 pages/s", "network path: ~120 new conn/s", BLUE),
                ("embed", "159", "600 passages/s", "GPU saturated", TEAL),
                ("index", "", "0.991 recall", "ef 128 buys +0.003 for 2× time", TEAL),
                ("query", "9.2 ms", "1.94 ms", "1.7 ms is the encoder itself", YEL)]
        g = VGroup()
        for name, a, b, ceil, c in rows:
            lab = T(name, 28, INK, weight="MEDIUM")
            track = Line(ORIGIN, RIGHT * 5.4, color=LINE, stroke_width=6)
            bar = Line(ORIGIN, RIGHT * 0.01, color=c, stroke_width=10)
            wall = DashedLine(UP * 0.3, DOWN * 0.3, color=RED, stroke_width=3).move_to(track.get_end())
            start = M(a, 18, MUTE)
            end = M(b, 22, c)
            cl = M(ceil, 18, MUTE)
            g.add(VGroup(lab, track, bar, wall, start, end, cl))
        y = 2.3
        anims = []
        for r in g:
            lab, track, bar, wall, start, end, cl = r
            lab.move_to([-6.4, y, 0], aligned_edge=LEFT)
            track.move_to([-4.7, y, 0], aligned_edge=LEFT)
            bar.move_to([-4.7, y, 0], aligned_edge=LEFT)
            wall.move_to(track.get_end())
            start.next_to(track.get_start(), DOWN, buff=0.15).shift(RIGHT * 0.1)
            end.next_to(wall, RIGHT, buff=0.25)
            cl.next_to(track, DOWN, buff=0.15).align_to(wall, RIGHT).shift(LEFT * 0.1)
            anims.append(AnimationGroup(FadeIn(lab), Create(track), FadeIn(start), Create(wall)))
            y -= 1.45
        self.go(LaggedStart(*anims, lag_ratio=0.2), run_time=self.t(0.2))
        grow = []
        for r in g:
            lab, track, bar, wall, start, end, cl = r
            grow.append(AnimationGroup(bar.animate.put_start_and_end_on(track.get_start(), track.get_end() - RIGHT * 0.05), FadeIn(end), FadeIn(cl)))
        self.go(LaggedStart(*grow, lag_ratio=0.25), run_time=self.t(0.4), rate_func=rate_functions.ease_out_cubic)
        lever = T("next lever: a smaller query encoder → training, for speed", 24, YEL).to_edge(DOWN, buff=0.6)
        self.go(FadeIn(lever), run_time=self.t(0.1))
        self.finish()

        self.narrate("n21")
        self.go(FadeOut(g), FadeOut(lever), run_time=self.t(0.1))
        pipe = pipeline(1.0)
        boxes, arrows = pipe
        self.go(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in boxes], *[GrowArrow(a) for a in arrows], lag_ratio=0.12), run_time=self.t(0.35))
        pulse = Dot(radius=0.12, color=YEL).move_to(boxes[0].get_center())
        pth = VMobject().set_points_as_corners([b.get_center() for b in boxes])
        self.go(MoveAlongPath(pulse, pth), run_time=self.t(0.3))
        name = T("Search", 64, INK, weight="BOLD").next_to(pipe, DOWN, buff=0.9)
        repo = M("github.com/amurthy04/Search", 22, MUTE).next_to(name, DOWN, buff=0.2)
        self.go(FadeOut(pulse), Write(name), FadeIn(repo), run_time=self.t(0.2))
        self.finish(1.5)
        self.go(FadeOut(Group(*self.mobjects)), run_time=1.0)
