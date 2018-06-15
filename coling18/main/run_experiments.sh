echo "Running ablation experiment" && cd ablation && sh run.sh && cd .. &&
echo "Running monolingual experiment" && cd monolingual && sh run.sh && cd .. &&
echo "Running crosslingual experiment" && cd multilingual && sh run.sh && cd .. &&
echo "Running lexicon creation" && cd lexicon_creation && run.sh && .. 

