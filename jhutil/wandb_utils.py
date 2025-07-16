import wandb
from wandb import Api
import plotly.graph_objects as go
from jhutil import color_log


def get_psnr_from_group(group_name, entity="jh11", project="GESI_diva360"):
    api = Api()
    runs = api.runs(f"{entity}/{project}", {"$and": [{"group": group_name}]}, per_page=1000)
    psnr = {}
    for run in runs:
        try:
            psnr[run.name] = run.summary["psnr"]
        except:
            continue

    return psnr


def show_psnr_diff(prev_group, new_group):

    psnr_prev = get_psnr_from_group(group_name=prev_group)
    psnr_new = get_psnr_from_group(group_name=new_group)

    psnr_prev = dict(sorted(psnr_prev.items(), key=lambda x: x[0]))
    psnr_new = dict(sorted(psnr_new.items(), key=lambda x: x[0]))
    categories = sorted(set(psnr_prev.keys()).union(set(psnr_new.keys())))
    differences = [psnr_new.get(cat, 0) - psnr_prev.get(cat, 0) for cat in categories]

    colors = ["blue" if diff < 0 else "red" for diff in differences]

    fig = go.Figure(data=[go.Bar(x=categories, y=differences, marker_color=colors)])
    fig.update_layout(
        title="PSNR Difference Bar Chart",
        xaxis_title="Category",
        yaxis_title="Difference (PSNR2 - PSNR1)",
        showlegend=False,
    )
    fig.show()


def get_avg_tracked_hours(
    group_name: str, entity: str = "jh11", project: str = "GESI_diva360",
):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group_name})

    runtimes = []
    for run in runs:
        try:
            runtime_sec = run.summary.get("_wandb", {}).get("runtime")
            if runtime_sec:
                runtimes.append(runtime_sec / 60)  # 초 → 시간
        except:
            continue

    if not runtimes:
        print("No runs found with tracked time.")
        return None

    avg_time = sum(runtimes) / len(runtimes)
    color_log(1111, f"Average Time for {group_name}: {avg_time:.2f}m")
    return avg_time
