import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ─────────────────────────────────────────────────────────────────────────────
# 1) Define terminal IDs + their flow percentages
# ─────────────────────────────────────────────────────────────────────────────
labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
percentages = [28, 14, 13, 25, 6, 6, 5, 1, 2]

# ─────────────────────────────────────────────────────────────────────────────
# 2) Create the figure & axes; leave extra bottom margin for images
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# We reserve about 40% of the figure’s height at the bottom for thumbnails
plt.subplots_adjust(bottom=0.40, top=0.90)

# Draw bars
bars = ax.bar(range(len(labels)), percentages, color="#33638D", edgecolor="black")

# Remove default x‐tick labels (we’ll place the images instead)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([])

ax.set_ylabel("Flow Percentage")
ax.set_title("Flow Percentage by Terminal Molecule")

# Extend the y‐limits downward to make sure negative‐y space is “live” (not clipped)
max_perc = max(percentages)
ax.set_ylim(-0.5 * max_perc, max_perc + 10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Parameters for consistent image height
#    We pick a “display height” in points (1 point = 1/72 inch).
#    All nine images will be scaled so that their drawn height = 80 points.
# ─────────────────────────────────────────────────────────────────────────────
desired_display_height_pts = 80  # each PNG will be drawn 80 points tall

# We will shift each image downward by half its height plus a little padding,
# in points, so that nothing overlaps the bars or gets cut.
vertical_offset_pts = -desired_display_height_pts / 2 - 10  # 10 points extra padding

# ─────────────────────────────────────────────────────────────────────────────
# 4) Loop over each label, load its PNG, compute a per‐image zoom factor
#    so that the final drawn height = desired_display_height_pts, then place it.
# ─────────────────────────────────────────────────────────────────────────────
for idx, label in enumerate(labels):
    filename = f"{label}.png"
    if not os.path.isfile(filename):
        print(f"Warning: '{filename}' not found – skipping that thumbnail.")
        continue

    # Read the image from disk
    arr_img = mpimg.imread(filename)
    img_h_pixels = arr_img.shape[0]  # native pixel height

    # Compute zoom: (native_pixel_height * zoom) in points → desired_display_height_pts
    zoom = desired_display_height_pts / img_h_pixels

    # Create an OffsetImage with that zoom
    imagebox = OffsetImage(arr_img, zoom=zoom)

    # Anchor at (x=idx, y=0), then offset downward by vertical_offset_pts points
    ab = AnnotationBbox(
        imagebox,
        (idx, 0),                # data‐coordinates (bar’s x, y=0)
        xycoords="data",
        boxcoords="offset points",
        xybox=(0, vertical_offset_pts),  # shift downward (pts)
        frameon=False,
        pad=0
    )
    ax.add_artist(ab)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Finally, show the plot
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.show()
