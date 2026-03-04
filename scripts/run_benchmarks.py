"""
Main benchmark comparison script.
Runs all routers on all datasets and generates comparison report.
"""

from sbscr.routers import SBSCRRouter, RandomRouter, KeywordRouter, SemanticRouter
from sbscr.evaluation import BenchmarkRunner
import json


def main():
    print("\n" + "="*70)
    print("SBSCR vs BASELINES - COMPREHENSIVE COMPARISON")
    print("="*70)
    
    # Initialize all routers
    routers = {
        'SBSCR': SBSCRRouter(registry_path="data/models.yaml", model_path="sbscr/models/complexity_xgboost.json"),
        'Random': RandomRouter(seed=42),
        'Keyword': KeywordRouter(),
        'Semantic': SemanticRouter(),
    }
    
    print("\n📊 Initialized Routers:")
    for name in routers.keys():
        print(f"  ✓ {name}")
    
    # Run benchmarks
    print("\n🔬 Running Benchmarks...")
    print("This may take a few moments (Semantic Router needs to load embeddings)...\n")
    
    runner = BenchmarkRunner(routers)
    results = runner.run_all_datasets(verbose=True)
    
    # Generate comparison report
    print("\n" + "="*70)
    report = runner.generate_comparison_report(results)
    print(report)
    
    # Save detailed results
    output_file = "benchmark_results.json"
    detailed_results = {}
    for dataset_name, dataset_results in results.items():
        detailed_results[dataset_name] = {}
        for router_name, metrics in dataset_results.items():
            detailed_results[dataset_name][router_name] = metrics.to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Compare SBSCR vs best baseline
    print("\n✨ SBSCR Performance:")
    
    for dataset_name, dataset_results in results.items():
        sbscr_metrics = dataset_results['SBSCR'].compute_metrics()
        semantic_metrics = dataset_results['Semantic'].compute_metrics()
        keyword_metrics = dataset_results['Keyword'].compute_metrics()
        
        print(f"\n  {dataset_name.upper()}:")
        print(f"    SBSCR Accuracy: {sbscr_metrics['accuracy']:.1%}")
        print(f"    SBSCR Latency: {sbscr_metrics['avg_latency_ms']:.2f}ms")
        print(f"    vs Semantic Accuracy: {semantic_metrics['accuracy']:.1%} "
              f"(Latency: {semantic_metrics['avg_latency_ms']:.2f}ms)")
        print(f"    vs Keyword Accuracy: {keyword_metrics['accuracy']:.1%} "
              f"(Latency: {keyword_metrics['avg_latency_ms']:.2f}ms)")
    
    print("\n" + "="*70)
    print("✅ Benchmark Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
