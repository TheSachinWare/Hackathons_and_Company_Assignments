from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def box(draw, xy, text, fill=(34, 45, 65)):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=(110, 129, 161), width=2)
    draw.text((x0 + 10, y0 + 12), text, fill=(230, 237, 243), font=FONT)


def arrow(draw, start, end, color=(110, 129, 161)):
    draw.line([start, end], fill=color, width=3)
    # arrow head
    hx, hy = end
    draw.polygon(
        [(hx, hy), (hx - 8, hy - 6), (hx - 8, hy + 6)],
        fill=color,
    )


WIDTH, HEIGHT = 900, 520
FONT = ImageFont.load_default()

if __name__ == "__main__":
    img = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 20))
    d = ImageDraw.Draw(img)

    box(d, (50, 200, 190, 260), "Binance WS Stream\n(trades)")
    box(d, (230, 170, 390, 230), "Ingestion\n(websocket -> queue)")
    box(d, (230, 250, 390, 310), "Upload CSV / OHLC")
    box(d, (430, 170, 620, 230), "Tick Store\n(thread-safe DataFrame)")
    box(d, (430, 250, 620, 310), "Resampling 1s/1m/5m\nRolling windows")
    box(d, (650, 170, 850, 230), "Analytics\n(hedge ratio, spread,\nz-score, ADF, corr)")
    box(d, (650, 260, 850, 330), "Alerts + Backtest\n(z-threshold, rules)")
    box(d, (430, 340, 850, 420), "Dash Frontend\n(controls, charts,\nexports)")

    arrow(d, (190, 230), (230, 200))
    arrow(d, (190, 230), (230, 280))
    arrow(d, (390, 200), (430, 200))
    arrow(d, (390, 280), (430, 280))
    arrow(d, (620, 200), (650, 200))
    arrow(d, (620, 280), (650, 290))
    arrow(d, (740, 230), (740, 260))
    arrow(d, (740, 330), (740, 340))

    out_path = Path(__file__).resolve().parent.parent / "architecture.png"
    img.save(out_path)
    print(f"Saved {out_path}")

