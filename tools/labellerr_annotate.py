"""Labellerr SDK wrapper: upload a dataset, stand up an annotation project, and
export accepted annotations as COCO JSON.

Mirrors the "Annotate your Custom dataset using Labellerr" section of
Fine_tune_yolo_for_football_analytics.ipynb, but driven by CLI args and .env
instead of notebook widgets/input() prompts.

Usage:
    python tools/labellerr_annotate.py setup \
        --folder D:/data/football_frames --dataset-name "My Dataset" \
        --project-name "My Project" --template-name "My Template" \
        --question Object --data-type image

    python tools/labellerr_annotate.py export \
        --project-id <id> --export-name "Weekly Export" \
        --export-format coco_json --statuses accepted
"""
import argparse
import os
import uuid

from dotenv import load_dotenv

load_dotenv()


def get_client():
    from labellerr.client import LabellerrClient

    try:
        api_key = os.environ["LABELLERR_API_KEY"]
        api_secret = os.environ["LABELLERR_API_SECRET"]
        client_id = os.environ["LABELLERR_CLIENT_ID"]
    except KeyError as exc:
        raise SystemExit(f"Missing {exc} in environment/.env") from exc

    return LabellerrClient(api_key, api_secret, client_id)


def setup_project(folder, dataset_name, project_name, template_name, question, data_type):
    from labellerr.core.annotation_templates import create_template
    from labellerr.core.datasets import create_dataset_from_local
    from labellerr.core.projects import create_project
    from labellerr.core.schemas import (
        AnnotationQuestion,
        CreateProjectParams,
        CreateTemplateParams,
        DatasetConfig,
        DatasetDataType,
        QuestionType,
        RotationConfig,
    )

    client = get_client()
    data_type_enum = DatasetDataType[data_type]

    dataset = create_dataset_from_local(
        client=client,
        dataset_config=DatasetConfig(dataset_name=dataset_name, data_type=data_type),
        folder_to_upload=folder,
    )
    print(f"Dataset created with ID: {dataset.dataset_id}")

    template = create_template(
        client=client,
        params=CreateTemplateParams(
            template_name=template_name,
            data_type=data_type_enum,
            questions=[
                AnnotationQuestion(
                    question_number=1,
                    question=question,
                    question_id=str(uuid.uuid4()),
                    question_type=QuestionType.polygon,
                    required=True,
                    color="#FF0000",
                )
            ],
        ),
    )
    print(f"Annotation template created with ID: {template.annotation_template_id}")

    dataset.status()  # wait until dataset is processed before creating the project

    project = create_project(
        client=client,
        params=CreateProjectParams(
            project_name=project_name,
            data_type=data_type_enum,
            rotations=RotationConfig(
                annotation_rotation_count=1,
                review_rotation_count=1,
                client_review_rotation_count=1,
            ),
        ),
        datasets=[dataset],
        annotation_template=template,
    )
    print(f"Project created: {project.project_id}")
    return project.project_id


def export_annotations(project_id, export_name, export_format, statuses):
    from labellerr.core.exceptions import LabellerrError
    from labellerr.core.projects import LabellerrProject

    client = get_client()
    project = LabellerrProject(client=client, project_id=project_id)

    export_config = {
        "export_name": export_name,
        "export_description": f"Export of {', '.join(statuses)} annotations",
        "export_format": export_format,
        "statuses": statuses,
    }

    try:
        result = project.create_local_export(export_config)
        export_id = result["response"]["report_id"]
        print(f"Local export created. Export ID: {export_id}")
    except LabellerrError as exc:
        raise SystemExit(f"Export creation failed: {exc}") from exc

    try:
        download_url = client.fetch_download_url(
            project_id=project_id, uuid=str(uuid.uuid4()), export_id=export_id
        )
        print(f"Download URL: {download_url}")
        return download_url
    except LabellerrError as exc:
        raise SystemExit(f"Failed to fetch download URL: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Upload a dataset and create an annotation project")
    setup_parser.add_argument("--folder", required=True, help="Local folder to upload as the dataset")
    setup_parser.add_argument("--dataset-name", required=True)
    setup_parser.add_argument("--project-name", required=True)
    setup_parser.add_argument("--template-name", required=True)
    setup_parser.add_argument("--question", default="Object", help="Annotation question label")
    setup_parser.add_argument("--data-type", default="image", choices=["image", "video"])

    export_parser = subparsers.add_parser("export", help="Export accepted annotations as COCO JSON")
    export_parser.add_argument("--project-id", required=True)
    export_parser.add_argument("--export-name", default="Export")
    export_parser.add_argument("--export-format", default="coco_json")
    export_parser.add_argument(
        "--statuses",
        nargs="+",
        default=["accepted"],
        help="Annotation statuses to include, e.g. review r_assigned client_review cr_assigned accepted",
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_project(
            folder=args.folder,
            dataset_name=args.dataset_name,
            project_name=args.project_name,
            template_name=args.template_name,
            question=args.question,
            data_type=args.data_type,
        )
    elif args.command == "export":
        export_annotations(
            project_id=args.project_id,
            export_name=args.export_name,
            export_format=args.export_format,
            statuses=args.statuses,
        )


if __name__ == "__main__":
    main()
