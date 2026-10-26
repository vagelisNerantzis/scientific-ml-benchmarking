"""Generate the public workflow SVG deterministically, without dependencies.

Run from the repository root: python assets/generate_workflow.py
Check the committed rendering: python assets/generate_workflow.py --check
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path

STEPS = [
    (40, 110, 'Scientific dataset', '', 'Repeated observations; multiple modalities'),
    (410, 110, 'Physical-sample', 'membership', 'Immutable identity, labels and partitions'),
    (780, 110, 'Train / validation /', 'locked test', 'Fit / develop / explicitly gated evaluation'),
    (780, 295, 'Model evaluation', '', 'Fixed protocol and declared seed set'),
    (410, 295, 'Physical-sample', 'aggregation', 'Pool observations before final metrics'),
    (40, 295, 'Metrics and calibration', '', 'Classification quality and uncertainty'),
    (40, 480, 'Repeated-seed', 'aggregation', 'Mean and sample standard deviation'),
    (410, 480, 'Physical-sample bootstrap', '/ paired inference', 'Keep seed replicates attached to samples'),
    (780, 480, 'Reproducible report', '', 'Protocol, hashes, environment and results'),
]


def render_svg():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="660" viewBox="0 0 1120 660" role="img" aria-labelledby="title description">',
             '<title id="title">Sample-aware scientific ML benchmarking workflow</title>',
             '<desc id="description">Scientific dataset to physical-sample membership to train, validation and locked test; model evaluation; physical-sample aggregation; metrics and calibration; repeated-seed aggregation; physical-sample bootstrap and paired inference; reproducible report. Follow numbered steps in a serpentine path.</desc>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
             '<rect width="1120" height="660" rx="20" fill="#f8fafc"/>',
             '<g font-family="sans-serif" fill="#0f172a">',
             '<text x="40" y="44" font-size="25" font-weight="700">Scientific ML Benchmarking</text>',
             '<text x="40" y="73" font-size="15" fill="#475569">Physical identity and provenance connect every stage of the experiment.</text>']
    for i, (x, y, line1, line2, caption) in enumerate(STEPS):
        color = '#0f766e' if i in (1, 7, 8) else '#2563eb'
        parts.extend([
            f'<rect x="{x}" y="{y}" width="300" height="122" rx="12" fill="white" stroke="#cbd5e1"/>',
            f'<rect x="{x}" y="{y + 12}" width="4" height="98" rx="2" fill="{color}"/>',
            f'<text x="{x + 18}" y="{y + 24}" font-size="12" font-weight="700" fill="{color}">{i + 1:02d}</text>',
            f'<text x="{x + 18}" y="{y + 52}" font-size="18" font-weight="600">{escape(line1)}</text>',
            f'<text x="{x + 18}" y="{y + 75}" font-size="18" font-weight="600">{escape(line2)}</text>',
            f'<text x="{x + 18}" y="{y + 103}" font-size="11.5" fill="#475569">{escape(caption)}</text>',
        ])
    arrows = [(348,171,400,171), (718,171,770,171), (930,240,930,285),
              (772,356,720,356), (402,356,350,356), (190,425,190,470),
              (348,541,400,541), (718,541,770,541)]
    for x1,y1,x2,y2 in arrows:
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    parts.extend(['<text x="40" y="637" font-size="13" fill="#475569">Independence follows the physical sample. Validation guides development. Test access is gated.</text>',
                  '</g>', '</svg>'])
    return '\n'.join(parts) + '\n'


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = Path(__file__).with_name('workflow.svg')
    svg = render_svg()
    if args.check:
        if not output.exists() or output.read_bytes() != svg.encode('utf-8'):
            parser.exit(1, 'Workflow SVG differs; regenerate with python assets/generate_workflow.py\n')
        print('Workflow SVG matches generator.')
    else:
        output.write_bytes(svg.encode('utf-8'))
        print('Generated workflow.svg')


if __name__ == '__main__':
    main()
