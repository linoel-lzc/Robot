import argparse
import sys
from robotarm.performer import run_performer
from robotarm.observer import run_observer

def main():
    parser = argparse.ArgumentParser(description="RobotArm CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Perform command
    perform_parser = subparsers.add_parser("perform", help="Run a melody sequence")
    perform_parser.add_argument("-m", "--melody", default="dial_8890.yaml", help="Score configuration file (default: dial_8890.yaml)")
    perform_parser.add_argument("--no-record", action="store_true", help="Disable video recording")

    # Observe command
    observe_parser = subparsers.add_parser("observe", help="Analyze video for key events")
    observe_parser.add_argument("video_file", help="Video file to analyze")
    observe_parser.add_argument("-m", "--melody", default="dial_8890.yaml", help="Score configuration file (default: dial_8890.yaml)")

    # Parse initial args to find subcommand
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()

    if args.command == "perform":
        # We need to reconstruct args for run_performer since it expects its own argparse
        # But wait, run_performer uses argparse inside it. 
        # A better way is to pass the specific args we parsed, 
        # but run_performer parses its own args from sys.argv or a list.
        # Let's pass the list of arguments relevant to the subcommand.
        
        cmd_args = []
        if args.melody:
            cmd_args.extend(["--melody", args.melody])
        if args.no_record:
            cmd_args.append("--no-record")
            
        sys.exit(run_performer(cmd_args))

    elif args.command == "observe":
        cmd_args = [args.video_file]
        if args.melody:
            cmd_args.extend(["--melody", args.melody])
            
        sys.exit(run_observer(cmd_args))
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

