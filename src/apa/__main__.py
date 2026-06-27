"""
Command-line interface for the APA framework.

    python -m apa diagnose "brooooooklyn/image"     # diagnose a stored case by repo/run_id
    python -m apa diagnose --file run.json          # diagnose a raw GitHub Actions record
    python -m apa list                              # list some available stored cases
    python -m apa diagnose "<query>" --json         # machine-readable output
"""
import argparse, json, sys
from .framework import diagnose, diagnose_raw, load_case, list_cases


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apa", description="Agentic CI failure diagnosis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="diagnose a CI failure")
    d.add_argument("query", nargs="?", help="repo or run_id substring of a stored case")
    d.add_argument("--file", help="path to a raw GitHub Actions / GHALogs JSON record")
    d.add_argument("--model", help="override the diagnostic model (else CI_AGENT_MODEL)")
    d.add_argument("--no-fix", action="store_true", help="skip the fix recommendation")
    d.add_argument("--json", action="store_true", help="emit JSON instead of the pretty trace")

    sub.add_parser("list", help="list some available stored cases")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        for repo in list_cases():
            print(" ", repo)
        return 0

    if args.cmd == "diagnose":
        if args.file:
            raw = json.load(open(args.file, encoding="utf-8"))
            dx = diagnose_raw(raw, model=args.model)
        elif args.query:
            case = load_case(args.query)
            dx = diagnose(case, model=args.model, recommend=not args.no_fix)
        else:
            ap.error("provide a stored-case query or --file")
        print(json.dumps(dx.to_dict(), indent=2) if args.json else dx.pretty())
        return 0


if __name__ == "__main__":
    sys.exit(main())
