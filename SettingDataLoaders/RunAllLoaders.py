"""
Run All Loaders - Execute all data loaders in sequence
"""

import os
import importlib.util
import sys

# List of loader modules to run
loader_modules = [
    'RaceLoader',
    'CharacterLoader',
    'FactionLoader',
    'LocationLoader',
    'SystemLoader',
    'ItemLoader',
    'CreatureLoader',
    'EventLoader',
    'ConceptLoader'
]


def run_loader(module_name):
    """Run a specific loader module by calling its main logic"""
    try:
        print(f"\n{'=' * 60}")
        print(f"Running {module_name}...")
        print('=' * 60)

        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Create the loader instance and run it
        # Get the loader class (e.g., CharacterLoader from CharacterLoader module)
        loader_class = getattr(module, module_name)

        # Determine the JSON file path based on data type
        # Convert module name to data type: CharacterLoader -> character
        data_type = module_name.lower().replace('loader', '')
        json_file = f"../SettingRawDataJSON/vanilla_fantasy/{data_type}s.json"

        # Create an instance and run load_and_insert
        loader = loader_class(json_file)
        result = loader.load_and_insert()

        # Print results
        print(f"Total records: {result.total_records}")
        print(f"Successfully inserted: {result.successful_inserts}")

        if result.errors:
            print(f"Errors encountered: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")

        return True

    except Exception as e:
        print(f"Error running {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all loader modules"""
    print("Starting all world data loaders...")
    print("=" * 60)

    # Change to the directory where the loaders are located
    original_dir = os.getcwd()
    loader_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        os.chdir(loader_dir)
        print(f"Running from directory: {loader_dir}")

        # Run each loader
        results = {}
        for module_name in loader_modules:
            if os.path.exists(f"{module_name}.py"):
                success = run_loader(module_name)
                results[module_name] = "Success" if success else "Failed"
            else:
                print(f"\n{module_name}.py not found, skipping...")
                results[module_name] = "Not Found"

    finally:
        os.chdir(original_dir)  # Change back to original directory

    # Print summary
    print("\n" + "=" * 60)
    print("LOADER EXECUTION SUMMARY")
    print("=" * 60)

    for module_name, status in results.items():
        print(f"{module_name:20} : {status}")

    # Count successes
    success_count = sum(1 for status in results.values() if status == "Success")
    total_count = len(results)

    print(f"\nSuccessfully completed: {success_count}/{total_count} loaders")

    if success_count == total_count:
        print("\n✅ All loaders completed successfully!")
    else:
        print(f"\n⚠️  {total_count - success_count} loader(s) failed or were not found")


if __name__ == "__main__":
    main()