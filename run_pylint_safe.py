#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para ejecutar Pylint con manejo seguro de Unicode
"""
import subprocess
import sys

def run_pylint_safe():
    """Ejecuta Pylint y guarda el resultado en UTF-8"""
    try:
        # Ejecutar pylint
        result = subprocess.run(
            [sys.executable, '-m', 'pylint', 'src', 'main.py', '--rcfile=.pylintrc'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # Reemplazar caracteres problemáticos
        )

        # Guardar resultado en archivo UTF-8
        output_file = 'reports/pylint_report_complete.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== ERRORES ===\n")
                f.write(result.stderr)

        print(f"[OK] Reporte guardado en: {output_file}")

        # Mostrar últimas líneas
        lines = result.stdout.split('\n')
        print("\n=== ULTIMAS 10 LINEAS ===")
        for line in lines[-10:]:
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode('ascii', 'replace').decode('ascii'))

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_pylint_safe())
