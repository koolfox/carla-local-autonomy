"""Illustrative workshop animations, not simulator footage or model output.

Render: manim -qh --fps 24 --media_dir /tmp/carla-manim vision_scenes.py FrameJourney PerceptionTasks
Manim Community 0.20.1. Shapes and numbers are schematic examples only.
"""

from manim import (
    DOWN,
    LEFT,
    UP,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    RoundedRectangle,
    Scene,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
)

BG = "#f5f5f8"
INK = "#172339"
MUTED = "#394b66"
BLUE = "#245dc4"
GOLD = "#a76818"
GREEN = "#207252"


def label(text, size=30, color=INK):
    return Text(text, font="Avenir Next", font_size=size, color=color)


class FrameJourney(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = label("From a frame to an observation", 40).to_edge(UP, buff=0.5)
        subtitle = label(
            "Conceptual illustration  •  not recorded CARLA footage", 20, MUTED
        ).to_edge(DOWN, buff=0.32)
        self.add(title, subtitle)
        steps = [
            ("CAMERA", "frame + time"),
            ("PREPROCESS", "resize / color"),
            ("MODEL", "estimate"),
            ("VIEW / RECORD", "frame + output"),
        ]
        boxes = VGroup()
        for i, (name, detail) in enumerate(steps):
            x, y = [(-3.3, 1.1), (3.3, 1.1), (3.3, -1.0), (-3.3, -1.0)][i]
            box = RoundedRectangle(
                width=5.0,
                height=1.35,
                corner_radius=0.13,
                color="#bfcbe0",
                fill_color="#ffffff",
                fill_opacity=1,
            ).move_to([x, y, 0])
            words = (
                VGroup(label(name, 30), label(detail, 24, MUTED))
                .arrange(DOWN, buff=0.2)
                .move_to(box)
            )
            boxes.add(VGroup(box, words))
        arrows = VGroup(
            Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.12, color=BLUE, stroke_width=3),
            Arrow(boxes[1].get_bottom(), boxes[2].get_top(), buff=0.1, color=BLUE, stroke_width=3),
            Arrow(boxes[2].get_left(), boxes[3].get_right(), buff=0.12, color=BLUE, stroke_width=3),
        )
        packet = Dot(color=BLUE, radius=0.09).move_to(boxes[0].get_bottom() + DOWN * 0.3)
        for i, box in enumerate(boxes):
            if i:
                self.play(Create(arrows[i - 1]), run_time=0.45)
            self.play(FadeIn(box, shift=UP * 0.12), run_time=0.6)
            if i == 0:
                self.play(FadeIn(packet), run_time=0.2)
            else:
                self.play(packet.animate.move_to(box.get_bottom() + DOWN * 0.3), run_time=0.5)
            self.play(Indicate(box[0], color=BLUE, scale_factor=1.03), run_time=0.55)
        warning = label(
            "Timing matters: a delayed result can describe an older frame.", 23, GOLD
        ).move_to([0, -2.3, 0])
        self.play(FadeIn(warning), run_time=0.7)
        self.wait(2.2)


class PerceptionTasks(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = label("One input, different questions", 40).to_edge(UP, buff=0.5)
        subtitle = label("Schematic example  •  not a model prediction", 20, MUTED).to_edge(
            DOWN, buff=0.32
        )
        self.add(title, subtitle)
        grid = VGroup(
            *[
                Square(
                    side_length=0.55,
                    stroke_color="#ffffff",
                    stroke_width=2,
                    fill_color=["#a9bfd9", "#c2d3df", "#dbe4df", "#9eb7ca"][(i * 3 + j) % 4],
                    fill_opacity=1,
                ).move_to([-4.6 + j * 0.55, 1.1 - i * 0.55, 0])
                for i in range(5)
                for j in range(5)
            ]
        )
        frame = label("IMAGE", 25, MUTED).next_to(grid, UP, buff=0.3)
        self.play(FadeIn(grid), FadeIn(frame), run_time=1)
        outputs = [
            ("DETECTION", "Where is the object?", BLUE),
            ("SEGMENTATION", "Which region is road?", GREEN),
            ("DEPTH", "How far might it be?", GOLD),
        ]
        rows = VGroup()
        for i, (head, question, color) in enumerate(outputs):
            bg = RoundedRectangle(
                width=6.8,
                height=0.92,
                corner_radius=0.12,
                color="#bfcbe0",
                fill_color="#ffffff",
                fill_opacity=1,
            )
            text = (
                VGroup(label(head, 23, color), label(question, 20, MUTED))
                .arrange(DOWN, buff=0.12)
                .move_to(bg)
            )
            rows.add(VGroup(bg, text).move_to([2.35, 1.35 - i * 1.18, 0]))
        for index, row in enumerate(rows):
            self.play(FadeIn(row, shift=LEFT * 0.2), run_time=0.7)
            if index == 0:
                box = SurroundingRectangle(
                    VGroup(grid[6], grid[7], grid[11], grid[12]),
                    color=BLUE,
                    buff=0.03,
                    stroke_width=5,
                )
                self.play(Create(box), run_time=0.5)
            elif index == 1:
                self.play(
                    FadeOut(box),
                    *[cell.animate.set_fill(GREEN, opacity=0.75) for cell in grid[15:]],
                    run_time=0.5,
                )
            else:
                palette = ["#d7e5f1", "#afc9df", "#7daed2", "#578fc1", "#3b70aa"]
                self.play(
                    *[
                        cell.animate.set_fill(palette[i // 5], opacity=1)
                        for i, cell in enumerate(grid)
                    ],
                    run_time=0.5,
                )
            self.wait(0.35)
        caveat = label("A prediction is not ground truth or a driving decision.", 23, GOLD).move_to(
            [0, -2.2, 0]
        )
        self.play(FadeIn(caveat), run_time=0.7)
        self.wait(2.2)
