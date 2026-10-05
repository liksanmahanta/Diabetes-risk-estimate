from diabetes_model import ARTIFACT_PATH, train_and_save


if __name__ == "__main__":
    bundle = train_and_save()
    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Selected by 5-fold training CV F1: {bundle['selected_model']}")
    print("\nHoldout metrics:")
    print(f"{'Model':<25} {'CV F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for model_name, scores in bundle["metrics"].items():
        print(
            f"{model_name:<25} {bundle['selection_cv_f1'][model_name]:>10.3f} "
            f"{scores['accuracy']:>10.3f} {scores['precision']:>10.3f} {scores['recall']:>10.3f} "
            f"{scores['f1']:>10.3f}"
        )