"""
运行Agent方法对比实验的便捷脚本
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """设置实验环境"""
    # 设置必要的环境变量
    os.environ.setdefault("DASHSCOPE_API_KEY", "sk-b773947f621d49dc949b5cd65e0f1340")
    os.environ.setdefault("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 创建必要的目录
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("agents", exist_ok=True)

def create_unified_experiment_folder():
    """创建统一的实验文件夹"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_folder = f"experiments/unified_comparison_{timestamp}"
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

def run_quick_comparison():
    """运行快速比较测试"""
    print("快速比较测试 - Hypothetical Minds vs WarAgent vs Werewolf vs Cognitive")
    print("="*50)
    
    try:
        from comparative_cognitive_world import ComparativeCognitiveWorld

        # 运行四种方法的比较 - 用 hm 替换 cot
        methods = ["cognitive", "hm", "war", "werewolf"]

        # 创建统一的实验文件夹
        unified_folder = create_unified_experiment_folder()
        print(f"快速对比结果将保存在: {unified_folder}")
        
        results = {}
        
        for method in methods:
            print(f"\n开始测试: {method}")
            try:
                if method == "cognitive":
                    result = run_cognitive_model_test(unified_folder)
                    if result:
                        results[method] = {
                            "scores": {"final_score": result.final_score},
                            "method_description": "认知增强模型 (RISE)"
                        }
                    else:
                        results[method] = {"error": "认知模型测试失败"}
                else:
                    result = run_single_method_test(method, unified_folder)
                    if result:
                        results[method] = {
                            "scores": {"final_score": result.final_score},
                            "method_description": f"{method.upper()} Agent"
                        }
                    else:
                        results[method] = {"error": f"{method}测试失败"}
            except Exception as e:
                results[method] = {"error": str(e)}
        
        # 简化结果显示
        print("\n快速对比结果:")
        print(f"{'方法':<15} {'最终得分':<10} {'状态'}")
        print("-" * 35)
        
        for method, result in results.items():
            if "error" in result:
                print(f"{method:<15} {'失败':<10} {result['error'][:15]}...")
            else:
                score = result["scores"]["final_score"]
                print(f"{method:<15} {score:<10.3f} {'成功'}")
        
        # 保存结果
        comparison_file = os.path.join(unified_folder, "quick_comparison_results.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n快速对比结果已保存到: {unified_folder}")
        return results
        
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保所有依赖文件都存在")
        return None
    except Exception as e:
        print(f"运行失败: {e}")
        return None

def run_cognitive_model_test(unified_folder: str = None, ablation_mode: str = "none"):
    """运行认知模型测试"""
    print("认知模型测试")
    print("="*30)
    
    try:
        # 导入认知模型
        from simulation.powergame.cognitive_world import CognitiveWorld
        
        print("开始运行认知模型测试...")
        if ablation_mode and ablation_mode != "none":
            print(f"使用消融模式: {ablation_mode}")
        timestamp = datetime.now().strftime("%m%d_%H%M")

        # 创建实验名称
        suffix = f"_{ablation_mode}" if ablation_mode and ablation_mode != "none" else ""
        experiment_name = f"cognitive_model_test{suffix}_{timestamp}"
        
        # 根据是否有统一文件夹决定base_dir
        if unified_folder:
            base_dir = unified_folder
        else:
            base_dir = "./experiments"
        
        # 创建认知世界实例
        cognitive_world = CognitiveWorld(
            experiment_name=experiment_name,
            use_rule_based=True,
            base_dir=base_dir,
            ablation_mode=ablation_mode
        )
        
        # 运行仿真
        cognitive_world.start_sim(max_steps=8)
        
        # 运行评测
        result = cognitive_world.run_final_evaluation()
        
        if result:
            print(f"\n认知模型测试结果:")
            print(f"历史事件对齐度 (EA): {result.ea_score:.3f}")
            print(f"行动内容相似度 (AS): {result.as_score:.3f}")
            print(f"战略合理性 (SR): {result.sr_score:.3f}")
            print(f"结果一致性 (OM): {result.om_score:.3f}")
            print(f"最终得分: {result.final_score:.3f}")
            print(f"实验结果保存在: {cognitive_world.experiment_logger.experiment_dir}")
            return result
        else:
            print("评测失败")
            return None
            
    except Exception as e:
        print(f"认知模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_cuban_ablation_comparison():
    """运行古巴导弹危机场景的消融对比（原模型+三次消融）"""
    print("古巴导弹危机 消融对比测试")
    print("="*50)
    unified_folder = create_unified_experiment_folder()
    print(f"所有对比结果将保存在: {unified_folder}")

    # 仅对比三种消融
    modes = {
        "no_world_profile": {"label": "消融：世界模型+侧写", "ablation": "no_world_profile"},
        "no_reasoning": {"label": "消融：假设推理", "ablation": "no_reasoning"},
        "no_all": {"label": "消融：全部" , "ablation": "no_all"}
    }

    results = {}
    for key, cfg in modes.items():
        print(f"\n开始测试: {cfg['label']} ({cfg['ablation']})")
        try:
            res = run_cognitive_model_test(unified_folder, ablation_mode=cfg['ablation'])
            if res:
                results[key] = {
                    "label": cfg['label'],
                    "ablation": cfg['ablation'],
                    "scores": {
                        "ea_score": res.ea_score,
                        "as_score": res.as_score,
                        "sr_score": res.sr_score,
                        "om_score": res.om_score,
                        "final_score": res.final_score
                    }
                }
            else:
                results[key] = {"label": cfg['label'], "ablation": cfg['ablation'], "error": "评测失败"}
        except Exception as e:
            results[key] = {"label": cfg['label'], "ablation": cfg['ablation'], "error": str(e)}

    # 输出对比
    print(f"\n{'方法':<18} {'最终':<8} {'EA':<8} {'AS':<8} {'SR':<8} {'OM':<8} 状态")
    print("-"*80)
    for key, item in results.items():
        label = item.get('label', key)
        if 'error' in item:
            print(f"{label:<18} {'失败':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {item['error'][:30]}...")
        else:
            s = item['scores']
            print(f"{label:<18} {s['final_score']:<8.3f} {s['ea_score']:<8.3f} {s['as_score']:<8.3f} {s['sr_score']:<8.3f} {s['om_score']:<8.3f} 成功")

    # 保存结果
    outfile = os.path.join(unified_folder, "cuban_ablation_comparison_results.json")
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {outfile}")
    return results


def run_cognitive_strategy_groups():
    """运行4组认知模型：两国采用相同策略（灵活/强硬/退让/以牙还牙）"""
    print("认知模型 四策略组对比（同策略双边）")
    print("="*50)

    try:
        from simulation.powergame.cognitive_world import CognitiveWorld
        from simulation.models.cognitive.country_strategy import (
            make_flexible_strategy,
            make_hardline_strategy,
            make_concession_strategy,
            make_tit_for_tat_strategy,
        )
    except Exception as e:
        print(f"导入失败: {e}")
        return None

    def _apply_strategy_to_both(world, factory):
        try:
            strat_a = factory()
            world.america.country_strategy = strat_a
            if hasattr(world.america, 'hypothesis_reasoning') and world.america.hypothesis_reasoning is not None:
                world.america.hypothesis_reasoning.country_strategy = strat_a
            strat_b = factory()
            world.soviet_union.country_strategy = strat_b
            if hasattr(world.soviet_union, 'hypothesis_reasoning') and world.soviet_union.hypothesis_reasoning is not None:
                world.soviet_union.hypothesis_reasoning.country_strategy = strat_b
        except Exception as e:
            print(f"设置策略失败: {e}")

    unified_folder = create_unified_experiment_folder()
    print(f"所有对比结果将保存在: {unified_folder}")

    strategies = [
        ("flexible", "灵活", make_flexible_strategy),
        ("hardline", "强硬", make_hardline_strategy),
        ("concession", "退让", make_concession_strategy),
        ("tit_for_tat", "以牙还牙", make_tit_for_tat_strategy),
    ]

    results = {}

    for key, label, factory in strategies:
        print(f"\n开始测试策略组：{label} ({key})")
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%m%d_%H%M")
            experiment_name = f"cog_strategy_{key}_{timestamp}"

            world = CognitiveWorld(
                experiment_name=experiment_name,
                use_rule_based=True,
                base_dir=unified_folder,
                ablation_mode="none",
            )

            _apply_strategy_to_both(world, factory)

            world.start_sim(max_steps=8)
            res = world.run_final_evaluation()

            if res:
                results[key] = {
                    "label": label,
                    "scores": {
                        "ea_score": res.ea_score,
                        "as_score": res.as_score,
                        "sr_score": res.sr_score,
                        "om_score": res.om_score,
                        "final_score": res.final_score,
                    },
                }
                print(f"{label} 最终得分: {res.final_score:.3f}")
            else:
                results[key] = {"label": label, "error": "评测失败"}
                print(f"{label} 评测失败")
        except Exception as e:
            results[key] = {"label": label, "error": str(e)}
            print(f"{label} 测试失败: {e}")

    print(f"\n{'方法':<10} {'最终':<8} {'EA':<8} {'AS':<8} {'SR':<8} {'OM':<8} 状态")
    print("-"*80)
    for key, item in results.items():
        label = item.get('label', key)
        if 'error' in item:
            print(f"{label:<10} {'失败':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {item['error'][:30]}...")
        else:
            s = item["scores"]
            print(f"{label:<10} {s['final_score']:<8.3f} {s['ea_score']:<8.3f} {s['as_score']:<8.3f} {s['sr_score']:<8.3f} {s['om_score']:<8.3f} 成功")

    outfile = os.path.join(unified_folder, "cognitive_strategy_groups_results.json")
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {outfile}")

    return results

def run_single_method_test(method: str = "cot", unified_folder: str = None):
    """运行单个方法测试"""
    print(f"单方法测试 - {method}")
    print("="*30)
    
    try:
        from comparative_cognitive_world import ComparativeCognitiveWorld
        
        # 创建实验名称，如果有统一文件夹则使用相对路径
        if unified_folder:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            experiment_name = f"{method}_test_{timestamp}"
            # 将实验保存到统一文件夹的子目录
            base_dir = unified_folder
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            experiment_name = f"{method}_test_{timestamp}"
            base_dir = "experiments"
        
        # 根据是否有统一文件夹决定base_dir
        if unified_folder:
            base_dir = unified_folder
        else:
            base_dir = "experiments"
        
        # 创建并运行实验
        world = ComparativeCognitiveWorld(
            agent_type=method,
            use_rule_based=True,
            experiment_name=experiment_name,
            base_dir=base_dir
        )
        
        print(f"开始运行 {method} 方法测试...")
        summary_file = world.start_sim(max_steps=8)
        
        print("运行评测...")
        result = world.run_final_evaluation()
        
        if result:
            print(f"\n{method} 测试结果:")
            print(f"历史事件对齐度 (EA): {result.ea_score:.3f}")
            print(f"行动内容相似度 (AS): {result.as_score:.3f}")
            print(f"战略合理性 (SR): {result.sr_score:.3f}")
            print(f"结果一致性 (OM): {result.om_score:.3f}")
            print(f"最终得分: {result.final_score:.3f}")
            print(f"实验结果保存在: {summary_file}")
            return result
        else:
            print("评测失败")
            return None
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_unified_comparison():
    """运行统一的对比测试，所有结果放在同一个文件夹"""
    print("统一对比测试")
    print("="*50)
    
    # 创建统一的实验文件夹
    unified_folder = create_unified_experiment_folder()
    print(f"所有对比结果将保存在: {unified_folder}")
    
    # 获取要测试的方法 - 用 hm 替换 cot
    print("\n可用方法: cognitive, hm, war, werewolf")
    methods_input = input("输入要比较的方法 (用逗号分隔, 如: cognitive,hm,war, 默认为全部): ").strip()

    if methods_input:
        methods = [m.strip() for m in methods_input.split(",") if m.strip() in ["cognitive", "hm", "war", "werewolf"]]
    else:
        methods = ["cognitive", "hm", "war", "werewolf"]

    print(f"将比较方法: {methods}")
    
    max_steps = input("最大步数 (默认8): ").strip()
    try:
        max_steps = int(max_steps) if max_steps else 8
    except ValueError:
        max_steps = 8
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"开始测试: {method}")
        print(f"{'='*60}")
        
        try:
            if method == "cognitive":
                # 使用认知模型，传递统一文件夹
                result = run_cognitive_model_test(unified_folder)
                if result:
                    results[method] = {
                        "scores": {
                            "ea_score": result.ea_score,
                            "as_score": result.as_score,
                            "sr_score": result.sr_score,
                            "om_score": result.om_score,
                            "final_score": result.final_score
                        },
                        "method_description": "认知增强模型 (RISE)"
                    }
                else:
                    results[method] = {"error": "认知模型测试失败"}
            else:
                # 使用其他方法
                result = run_single_method_test(method, unified_folder)
                if result:
                    results[method] = {
                        "scores": {
                            "ea_score": result.ea_score,
                            "as_score": result.as_score,
                            "sr_score": result.sr_score,
                            "om_score": result.om_score,
                            "final_score": result.final_score
                        },
                        "method_description": f"{method.upper()} Agent"
                    }
                else:
                    results[method] = {"error": f"{method}测试失败"}
                    
        except Exception as e:
            print(f"{method} 测试失败: {e}")
            results[method] = {"error": str(e)}
    
    # 输出对比结果
    print(f"\n{'='*80}")
    print("统一对比结果总结")
    print(f"{'='*80}")
    
    print(f"{'方法':<15} {'最终得分':<10} {'EA':<8} {'AS':<8} {'SR':<8} {'OM':<8} {'状态'}")
    print("-" * 80)
    
    for method, result in results.items():
        if "error" in result:
            print(f"{method:<15} {'失败':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {result['error'][:20]}...")
        else:
            scores = result["scores"]
            print(f"{method:<15} {scores['final_score']:<10.3f} "
                  f"{scores['ea_score']:<8.3f} {scores['as_score']:<8.3f} "
                  f"{scores['sr_score']:<8.3f} {scores['om_score']:<8.3f} {'成功'}")
    
    # 保存对比结果到统一文件夹
    comparison_file = os.path.join(unified_folder, "unified_comparison_results.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有对比结果已保存到: {unified_folder}")
    print(f"对比结果文件: {comparison_file}")
    
    return results

def main():
    """主函数"""
    print("Agent决策方法对比实验")
    print("="*50)
    
    # 设置环境
    setup_environment()
    
    # 选择运行模式
    print("\n运行模式选择:")
    print("1. 快速对比测试 (推荐) - 包含认知模型")
    print("2. 单方法测试")
    print("3. 统一对比测试 - 所有结果在同一文件夹")
    print("4. 认知模型独立测试")
    print("5. 自定义对比测试")
    print("6. 古巴导弹危机 消融对比（三次消融）")
    print("7. 认知模型 四策略组对比（两国同策）")
    
    choice = input("请选择运行模式 (1-7, 默认1): ").strip() or "1"
    
    if choice == "1":
        # 快速对比测试
        results = run_quick_comparison()
        if results:
            print("\n✓ 快速对比测试完成")
        else:
            print("\n✗ 快速对比测试失败")
    
    elif choice == "2":
        # 单方法测试
        print("\n可用方法: cognitive, hm, war, werewolf")
        method = input("选择方法 (默认hm): ").strip() or "hm"

        if method not in ["cognitive", "hm", "war", "werewolf"]:
            print("无效方法，使用默认的hm")
            method = "hm"

        if method == "cognitive":
            result = run_cognitive_model_test()
        else:
            result = run_single_method_test(method)
        
        if result:
            print(f"\n✓ {method} 测试完成")
        else:
            print(f"\n✗ {method} 测试失败")
    
    elif choice == "3":
        # 统一对比测试
        results = run_unified_comparison()
        if results:
            print("\n✓ 统一对比测试完成")
        else:
            print("\n✗ 统一对比测试失败")
    
    elif choice == "4":
        # 认知模型独立测试
        result = run_cognitive_model_test()
        if result:
            print("\n✓ 认知模型测试完成")
        else:
            print("\n✗ 认知模型测试失败")
    
    elif choice == "5":
        # 自定义对比测试
        print("\n可用方法: cognitive, hm, war, werewolf")
        methods_input = input("输入要比较的方法 (用逗号分隔, 如: cognitive,hm,war): ").strip()

        if methods_input:
            methods = [m.strip() for m in methods_input.split(",") if m.strip() in ["cognitive", "hm", "war", "werewolf"]]
        else:
            methods = ["hm", "war"]

        print(f"将比较方法: {methods}")
        
        max_steps = input("最大步数 (默认8): ").strip()
        try:
            max_steps = int(max_steps) if max_steps else 8
        except ValueError:
            max_steps = 8
        
        try:
            # 使用统一文件夹进行自定义对比
            results = run_unified_comparison()
            if results:
                print(f"\n✓ 自定义对比测试完成")
            else:
                print(f"\n✗ 自定义对比测试失败")
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
    
    else:
        if choice == "6":
            # 古巴导弹危机 消融对比
            results = run_cuban_ablation_comparison()
            if results:
                print("\n✓ 消融对比测试完成")
            else:
                print("\n✗ 消融对比测试失败")
        elif choice == "7":
            # 四策略组对比（两国同策）
            results = run_cognitive_strategy_groups()
            if results:
                print("\n✓ 四策略组对比完成")
            else:
                print("\n✗ 四策略组对比失败")
        else:
            print("无效选择")

if __name__ == "__main__":
    main()