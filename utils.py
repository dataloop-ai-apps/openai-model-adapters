import json
import os
import dtlpy as dl


def upload_llm_trace(dataset: dl.Dataset, trace_path="test_trace.json", remote_path="/llm_traces"):
    """Upload an LLMTrace item to a dataset and return the item."""
    item = dataset.items.upload(local_path=trace_path, remote_path=remote_path)
    print(f"Uploaded LLMTrace item: {item.id}  ({item.filename})")
    return item


def download_trace(item: dl.Item, output_path="result_trace.json"):
    """Download an LLMTrace item's JSON body to a local file."""
    trace = dl.LLMTrace.from_item(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace.to_json(), f, indent=2, ensure_ascii=False)
    print(f"Downloaded trace to {output_path}")
    return output_path


def publish_and_install(project: dl.Project, manifest_path="adapters/chat_completion/dataloop.json"):
    """Publish the adapter as a DPK and install it on the project."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    app_name = manifest["name"]
    print(f"Publishing {app_name} v{manifest['version']} to project {project.name}")

    dpk = dl.Dpk.from_json(manifest)
    dpk.codebase = project.codebases.pack(
        directory=os.getcwd(),
        name=dpk.display_name,
        extension="dpk",
        ignore_directories=[".venv", "venv", ".git", "__pycache__", "output"],
        ignore_max_file_size=True,
    )
    dpk = project.dpks.publish(dpk=dpk)
    print(f"Published DPK: {dpk.name} v{dpk.version} (id: {dpk.id})")

    try:
        app = project.apps.get(app_name=dpk.display_name)
        print("Already installed, updating...")
        app.dpk_version = dpk.version
        app.update()
        print(f"Updated app: {app.id}")
    except dl.exceptions.NotFound:
        print("Installing...")
        app = project.apps.install(dpk=dpk, app_name=dpk.display_name)
        print(f"Installed app: {app.id}")

    return app


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dl.setenv("prod")
    if dl.token_expired():
        dl.login()

    project = dl.projects.get(project_id="07438556-4b49-474f-9f4c-59b14e7781b8")
    dataset = project.datasets.get(dataset_id="69c7a3f357e0f68d28a51a58")

    # 1. Upload a test LLMTrace item
    # item = upload_llm_trace(dataset, trace_path="test_trace.json")
    # print(f"Uploaded item: {item.id}")
    item = dl.items.get(item_id="69c7a8ca9986f9db2631392a")
    download_trace(item, output_path="result_trace.json")

    # # 2. Deploy the adapter
    # app = publish_and_install(project)
