# How to build & test
cd to open_spiel root path
```bash
cd /path/to/open_spiel
export PYTHONPATH=`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/build/python
```

Run specific test
```bash
./open_spiel/scripts/build_and_run_tests.sh --virtualenv=false --test_only=chinese_chess_test
```

Run example
```bash
.build/examples/example --game=chinese_chess
```

Update Python integration tests
```bash
./open_spiel/scripts/generate_new_playthrough.sh chinese_chess
```

Run playthrough_test
```bash
python ./open_spiel/integration_tests/playthrough_test.py
```

# Implementation chinese_chess board

- Board definition
- Initial board from FEN
- Chinese Chess rules