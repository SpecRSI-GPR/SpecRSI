# SpecRSI: online-only supplementary materials

This supplement presents the released Drafter prompt template and a supplementary D+H csvcut case. [Main-paper results](README.md) use the manuscript's figure and table numbers. A [PDF version](online_supplement.pdf) is also available.

## Drafter prompt template

The released [HFPromptDrafter](../../src/specrsi/learning/bridge.py) uses the following template for task agents represented by a single system instruction. Braced fields are filled at runtime.

<div style="border:1px solid #404040;border-radius:8px;overflow:hidden;margin:16px 0;color:#000;background:#F2F2F2;">
<div style="background:#404040;color:#FFF;font-family:Georgia,serif;font-size:1.15em;padding:10px 16px;">Prompt O1. Drafter for task-agent instructions</div>
<div style="padding:14px 16px;line-height:1.5;">
<p style="margin-top:0;">Write an improved system instruction for the task agent. Return only the new instruction, without commentary. Make a concrete change using the observed feedback; do not copy the current instruction. Preserve the required output format and interface constraints in the current instruction. The checker and evaluation rules cannot be edited.</p>
<p><code>{interface_guidance}</code></p>
<p><strong>Current instruction:</strong><br><code>{bounded_parent}</code></p>
<p style="margin-bottom:0;"><strong>Observed search context:</strong><br><code>{context}</code></p>
</div>
</div>

The [feedback serializer](../../src/specrsi/learning/draft_context.py) includes up to four parent-task examples, with visible task inputs and observed scores, admissibility, and available answer/checker feedback. A null outcome denotes an unobserved result. Examples are selected in failure-first order; task references and metadata are not read. The current instruction and serialized feedback are bounded by the available input-token allowance. The box reproduces the released implementation template, rather than a logged prompt from the case below.

<a id="table-o2-dh-csvcut-case"></a>

## Table O2. D+H csvcut case

Costs and token shares summarize five seeds.

<table style="border-collapse:collapse;width:100%;color:#000;border-top:2px solid #000;border-bottom:2px solid #000;">
<tbody>
<tr style="border-bottom:1px solid #000;"><th scope="row" style="text-align:left;vertical-align:top;padding:8px;white-space:nowrap;">Setup</th><td style="padding:8px;"><span style="color:#2F65B0;">MetaRSI-H and SpecRSI-H add <code>--skip-header</code> to <code>csvcut</code> while preserving compatibility with <code>--columns</code> and <code>--out-delimiter</code>. Both use the same root, D+H permissions, and a 2.0-k-token instance budget; task-model weights remain frozen. The root fails all 12 new feature tests.</span></td></tr>
<tr style="border-bottom:1px solid #000;"><th scope="row" style="text-align:left;vertical-align:top;padding:8px;white-space:nowrap;">MetaRSI-H trace</th><td style="padding:8px;">Adds argument parsing and help text, then header-skipping logic. The first revision passes the original regression suite, <span style="color:#B43C3C;">while the 12 new feature tests still fail</span>. After header-skipping logic is added, all 12 new feature tests pass. The final combined suite passes, with clean static checks.</td></tr>
<tr style="border-bottom:1px solid #000;"><th scope="row" style="text-align:left;vertical-align:top;padding:8px;white-space:nowrap;">SpecRSI-H trace</th><td style="padding:8px;"><span style="color:#21804A;">Drafts argument parsing, header skipping, and empty/single-line handling with corresponding tests. Syntax and module checks precede the final full suite and static checks. The final version passes all 12 new feature tests and the original regression suite.</span></td></tr>
<tr style="border-bottom:1px solid #000;"><th scope="row" style="text-align:left;vertical-align:top;padding:8px;">Search cost<br>(five-seed means)</th><td style="padding:8px;">MetaRSI-H: <strong>1.722</strong> k tokens; SpecRSI-H: <span style="color:#21804A;"><strong>1.468</strong> k tokens</span>. MetaRSI-H token shares are 78% full-test interaction, 18% generation, and 4% other search inference. SpecRSI-H uses 42% on full-test interaction, 11% on staged checks, 35% on generation, and 12% on planning.</td></tr>
<tr><th scope="row" style="text-align:left;vertical-align:top;padding:8px;color:#B35C12;">Analysis</th><td style="padding:8px;color:#B35C12;">Both versions complete the feature. SpecRSI-H uses staged checks and reduces mean search-token cost by <strong>14.8%</strong>, with task-model weights frozen throughout.</td></tr>
</tbody>
</table>

Search cost counts forward input/output and tool-return text tokens; test execution is not converted to tokens. Blue marks the setup, red marks incomplete intermediate functionality, green marks the SpecRSI-H trace and cost, and orange marks the analysis.
