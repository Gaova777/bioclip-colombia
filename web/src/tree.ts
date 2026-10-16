// The same walk as domain/abstention.py, done in the browser so the slider doesn't need the server.

export const RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'] as const
export type Rank = (typeof RANKS)[number]

export interface TreeNode {
  rank: Rank
  path: string[]
  p: number
}

export type Classification = Record<string, TreeNode[]>

export interface Step {
  rank: Rank
  options: TreeNode[] // children of the node chosen one rank above, most likely first
  trusted: boolean // false from the first rank where the best option is below the threshold
}

const isChild = (node: TreeNode, parent: string[]) =>
  node.path.length === parent.length + 1 && parent.every((name, i) => node.path[i] === name)

// Keeps following the most likely child after the threshold is crossed,
// so the page can still show what the model would have said.
export function walk(nodes: TreeNode[], threshold: number): Step[] {
  const steps: Step[] = []
  let parent: string[] = []
  let trusted = true
  for (const rank of RANKS) {
    const options = nodes.filter((n) => n.rank === rank && isChild(n, parent)).sort((a, b) => b.p - a.p)
    trusted = trusted && options[0].p >= threshold
    steps.push({ rank, options, trusted })
    parent = options[0].path
  }
  return steps
}

export function displayName(node: TreeNode): string {
  return node.rank === 'species' ? node.path.slice(-2).join(' ') : node.path[node.path.length - 1]
}
