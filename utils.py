import os
import subprocess

from telemetry import traced_call


@traced_call(
    "tool",
    "latex.compile",
    metadata_factory=lambda args, kwargs: {
        "compile_pdf": kwargs.get("compile", args[2] if len(args) > 2 else True),
    },
    success_predicate=lambda result: "[CODE EXECUTION ERROR]" not in str(result),
)
def compile_latex(latex_code, output_path, compile=True, timeout=30):
    tex_dir = os.path.join(str(output_path), "tex")
    os.makedirs(tex_dir, exist_ok=True)
    tex_path = os.path.join(tex_dir, "temp.tex")
    with open(tex_path, "w", encoding="utf-8") as output:
        output.write(latex_code)
    if not compile:
        return "Compilation disabled; LaTeX source saved"
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=tex_dir,
        )
        return f"Compilation successful: {result.stdout.decode('utf-8', errors='replace')}"
    except subprocess.TimeoutExpired:
        return f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds"
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"[CODE EXECUTION ERROR]: Compilation failed: {exc}"
