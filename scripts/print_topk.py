# Trả về top n khoá học được đề xuất cho một sinh viên,
import argparse
from app import load_data, get_recommender, apply_recommendation_business_rules
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print top-K course recommendations for a student (same logic as UI)."
    )
    parser.add_argument("--student", required=True, help="Student ID, e.g. S121")
    parser.add_argument("-k", "--topk", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()

    student_id = args.student.strip().upper()
    k = max(1, int(args.topk))

    data = load_data()
    recommender = get_recommender()

    recs = recommender.recommend(student_id, n=k)
    recs = apply_recommendation_business_rules(student_id, recs, data, limit=k)

    if recs.empty:
        print(f"No recommendations for {student_id}")
        return

    print(f"Top {min(k, len(recs))} recommendations for {student_id}")
    for i, row in enumerate(recs.to_dict("records"), start=1):
        course_id = row.get("course_id", "")
        title = row.get("title", "")
        score = row.get("hybrid_score", 0.0)
        print(f"{i:2d}. {course_id} | {title} | hybrid_score={score:.4f}")


if __name__ == "__main__":
    main()

