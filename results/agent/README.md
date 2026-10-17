# The field guide agent: does asking questions help?

When BioCLIP stops before the species (343 of the 1,000 test photos at the 71% threshold), the agent asks
yes/no questions about what the person sees and updates the candidates' probabilities with the answers.

## How the questions are chosen

1. **Field marks per species, from photos.** A local vision-language model (Qwen3-VL 8B instruct, through Ollama)
   looked at the 5 validation photos of each of the 200 species and filled in a form with a closed vocabulary:
   colour of crown, throat, underparts, back and tail; bill colour and shape; tail shape. A mark is kept only if
   at least 3 of the 5 photos agree ([`data/field_marks.json`](../../data/field_marks.json), 7.2 of 8 marks per species
   on average). Photos that disagree (a female, a juvenile, bad light) leave the mark unknown, which is what it should be.
2. **The question with the most information.** For every (feature, value) pair, the expected information gain over
   the candidates, weighted by BioCLIP's probabilities. A feature all the likely candidates share is worth 0 bits
   and is never asked ([`domain/field_marks.py`](../../src/bioclip_co/domain/field_marks.py)).
3. **Bayes' rule with the answer.** A "yes" multiplies a candidate's probability by 0.9 if its mark matches, 0.1 if it
   doesn't, 0.5 if unknown; "can't tell" changes nothing. Then the same walk down the tree as the classifier decides
   whether it can name the species. At most 3 questions.

The language model doesn't choose questions or species. It only writes the final card from the cited Wikipedia text.

## Result

[`scripts/evaluate_agent.py`](../../scripts/evaluate_agent.py) runs the real LangGraph agent on those 343 photos with a
simulated person who answers from the true species' field marks ("can't tell" when the mark is unknown), sometimes
getting it wrong on purpose. No region filter, no network.

On the 343 photos where BioCLIP stops:

| simulated person | right species | wrong species | right branch, higher rank | wrong branch | questions |
|---|---:|---:|---:|---:|---:|
| always answers right | **60.6%** | 9.6% | 24.8% | 5.0% | 1.94 |
| wrong 10% of the time | 50.4% | 16.6% | 27.7% | 5.2% | 1.94 |
| wrong 20% of the time | 45.2% | 22.4% | 27.4% | 5.0% | 1.92 |

Over all 1,000 test photos:

| | right species | wrong answers |
|---|---:|---:|
| BioCLIP alone (71% threshold) | 62.8% | 4.1% |
| + agent, person always right | **83.6%** | 7.9% |
| + agent, person wrong 10% of the time | 80.1% | 10.4% |
| + agent, person wrong 20% of the time | 78.3% | 12.3% |

([`agent_eval.csv`](agent_eval.csv))

## What I take from it

- **About two questions turn six in ten unsure photos into the right species.** Over the whole test set that's
  21 more points of photos named correctly.
- **It's a trade, not a free win.** Wrong answers almost double (4.1% → 7.9%) even with a perfect person, and grow
  with a sloppy one. Once the agent commits to a species it is wrong 14% of the time, against 4.4% for BioCLIP's
  own species answers. A stricter threshold for the agent's last step would trade some of that back; I haven't
  tuned one, because the field marks come from the validation photos and tuning on them would be cheating.
- **This is the best case for the questions.** The simulated person agrees with the field guide by construction.
  A real person can misjudge a colour in the field, or see a female when the marks describe the male.
- **The ceiling is BioCLIP's shortlist.** The true species is among the agent's candidates in 79% of these photos;
  in the other 21% no question can find it.
- **Some birds look alike for everyone.** The yellow warbler and the saffron finch both come out all yellow in their
  field marks, the same pair BioCLIP confuses: no question in this vocabulary separates them.

## What didn't work

- **Letting the LLM write the questions.** My first agent had a 4B language model read the candidates' Wikipedia
  text and write a question. It sometimes asked about a feature every remaining candidate shared ("a violet crown?"
  when both had one), and, asked to pick among three species, it once picked the one the answer had just ruled out.
- **Field marks from text.** Then I had the model extract field marks from Wikipedia descriptions. Many descriptions
  don't give the colours (47 species had no description section at all), and the model filled the gaps by guessing:
  it gave the great kiskadee, famous for its yellow belly, a white one. A bigger 8B model guessed less but its
  "thinking" variant ignored the instruction not to think and returned broken JSON. Looking at photos replaced the text.
- Wikipedia throttled my first requests (HTTP 429): their API wants a user agent that says who you are.
