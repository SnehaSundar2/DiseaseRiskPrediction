"""Entry point for the Disease Risk Prediction Demo (a learning project on synthetic data, not medical advice).

Usage:
    python main.py                                       # interactive menu
    python main.py --train                               # train on generated sample people
    python main.py --train --data my_people.csv          # train on your own CSV (see README for columns)
    python main.py --predict age=58 sex=Male bmi=29 systolic_bp=145 smoker=yes
    python main.py --score sample_data/people_to_score.csv   # score a CSV of people
    python main.py --serve --port 8040                   # local web interface
    python main.py --report reports/model_report.html    # HTML report of the trained model
    python main.py --generate sample_data/people.csv --rows 3000
"""

import argparse
import logging
import sys
from pathlib import Path

from riskpredict import datagen, persistence, report, schema as S, training
from riskpredict import preprocessing as P
from riskpredict import validators as v
from riskpredict.cli import SYNTHETIC_ROWS, RiskCLI
from riskpredict.exceptions import RiskError
from riskpredict.predictor import require_predictor
from riskpredict.web import serve


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # show block characters and arrows on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Disease Risk Prediction Demo (machine learning project; not medical advice)")
    parser.add_argument("--model-dir", default="models", help="folder for the trained model (default models)")
    parser.add_argument("--train", action="store_true", help="train a model and save it")
    parser.add_argument("--data", metavar="CSV", help="with --train: your own dataset (default: generated sample data)")
    parser.add_argument("--rows", default=str(SYNTHETIC_ROWS), help="people to generate (default 3000)")
    parser.add_argument("--fast", action="store_true", help="with --train: smaller models, much faster")
    parser.add_argument("--no-tune", action="store_true", help="with --train: skip hyper-parameter tuning")
    parser.add_argument("--sensitivity", default=str(S.TARGET_SENSITIVITY), help="with --train: share of future cases the screening rule should catch (default 0.85)")
    parser.add_argument("--predict", nargs="+", metavar="KEY=VALUE", help="estimate one person, e.g. age=58 sex=Male bmi=29 systolic_bp=145")
    parser.add_argument("--score", metavar="CSV", help="score every person in a CSV file")
    parser.add_argument("--output", metavar="CSV", help="with --score: where to save the scores")
    parser.add_argument("--serve", action="store_true", help="start the local web interface")
    parser.add_argument("--port", default="8040", help="web interface port (default 8040)")
    parser.add_argument("--report", metavar="HTML", help="save an HTML report of the trained model")
    parser.add_argument("--generate", metavar="CSV", help="write a synthetic dataset to a CSV file")
    parser.add_argument("--clean-data", action="store_true", help="with --generate: a dataset without data problems")
    args = parser.parse_args(argv)

    logging.basicConfig(filename="riskpredict.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli = RiskCLI(args.model_dir)
    try:
        if args.generate:
            rows = v.validate_int(args.rows, "Rows", 300, 50000)
            path = Path(args.generate)
            if path.exists():
                print(f"Error: {path} already exists; choose another name.")
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            datagen.generate(rows, dirty=not args.clean_data).to_csv(path, index=False)
            print(f"Saved {rows} synthetic people to {path}.")
            return 0
        if args.train:
            target = v.validate_number(args.sensitivity, "Sensitivity", 0.5, 0.99)
            if args.data:
                df, label = P.load_dataset(args.data), Path(args.data).name
            else:
                df, label = datagen.generate(v.validate_int(args.rows, "Rows", 500, 50000)), "synthetic (generated)"
            result = training.train(df, fast=args.fast, tune=not (args.fast or args.no_tune), source=label,
                                    progress=lambda m: print(f"  {m}"), target_sensitivity=target)
            persistence.save(result, args.model_dir)
            print("\n" + "\n".join(result.cleaning.lines()) + "\n")
            print(report.comparison_text(result.metadata) + "\n")
            print(report.model_text(result.metadata))
            print(f"\nModel saved to {args.model_dir}/")
            return 0
        if args.predict:
            raw = {}
            for pair in args.predict:
                if "=" not in pair:
                    print(f"Error: '{pair}' should look like key=value, e.g. age=58.")
                    return 1
                key, value = pair.split("=", 1)
                raw[key] = value
            cli.show_prediction(require_predictor(args.model_dir).predict(raw))
            return 0
        if args.score:
            target = args.output or str(Path("predictions") / f"{Path(args.score).stem}_scored.csv")
            result = require_predictor(args.model_dir).predict_csv(args.score, target)
            failed = (result["status"] != "ok").sum()
            print(f"Scored {len(result) - failed} person(s) and saved them to {target}" + (f"; {failed} row(s) could not be scored." if failed else "."))
            return 0
        if args.report:
            print(f"Report saved to {report.save_html(require_predictor(args.model_dir).meta, args.report)}")
            return 0
        if args.serve:
            serve(require_predictor(args.model_dir), v.validate_port(args.port))
            return 0
        cli.run()
        return 0
    except RiskError as exc:
        print(f"Error: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
