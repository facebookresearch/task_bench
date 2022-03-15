# TaskBench500
The TaskBench500 dataset and code for generating tasks.

## Data
The TaskBench dataset is currently available under dl.fbaipublicfiles.com/task_bench/taskbench500.tar.gz

After downloading, expand with
```bash
tar -xzvf taskbench500.tar.gz
```

The file structure looks as follows:

    .
    ├── atomic                      # atomic tasks
    ├── seq_composite               # sequential compositional tasks
    │   ├── filter                  # tasks of the form filter{λx.F(x)}(S)
    │   ├── map                     # tasks of the form map{λx.F(x)}(S)
    │   └── mapfilter               # tasks of the form map{λx.F(x)}(filter{λx.F(x)}(S))
    └── word_composite              # word-level compositional tasks
        ├── chaining                # tasks of the form F(F')
        ├── intersection            # tasks of the form F∩F'
        ├── land                    # tasks of the form F∧F'
        ├── lor                     # tasks of the form F∨F'
        └── union                   # tasks of the form F∪F'

where `F` and `F'` are arbitrary word-level functions.

## Dataset Creation Procedure
Coming soon.

## Training a Model
Coming soon.

# Licensing
The majority of TaskBench500 is licensed under CC0, however portions of the project are available under separate license terms: nltk is licensed under the Apache 2.0 license; tqdm is licensed under the MIT license; and functools and glob are licensed under the BSD license.
