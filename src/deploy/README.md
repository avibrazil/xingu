# Releases de AVM

## Preparação de Artefatos de Produção

1. Comitê para decidir todas as features que vão entrar.
1. Criação da branch `release`.
1. Merge para `release` de todas as branches e features aprovadas.
1. Deixar ativos somente os DataProviders relevantes para a versão; comentar todos os outros.
1. Treino de todos os estimadores relevantes, no ambiente de staging (usando workflow [Train & Predict (✅ staging)](https://github.com/loft-br/robson_avm/actions/workflows/build_and_train_staging.yml)), e construção do `currents.yaml`, na branch `release`.
1. Teste de deploy dos estimadores prontos, usando a branch `release`, usando o comando `robson.deploy` para copiar PKLs e seus dados para um laptop ou SageMaker (exemplos na POC 8).
1. Mesmo teste, só que no Actions, usando o workflow [Deploy ✅Local Test from ✅Staging](https://github.com/loft-br/robson_avm/actions/workflows/deploy_localtest_from_staging.yml) contra branch `release`. Verifique o resultado do teste no final dos logs.

## Deploy da API - robson_avm (SageMaker)

1. Teste de criação do container da API, usando a branch `release`.
1. Teste de execução do container da API.
1. Merge `release` ➔ `main`.<br/>
   _Depois deste passo, cálculo semanal de métricas em produção passam a usar a nova release._
1. Rodar workflow [Deploy ⛔Production from ✅Staging](https://github.com/loft-br/robson_avm/actions/workflows/deploy_production_from_staging.yml) para publicar artefatos no ambiente de produção.
1. Rodar workflow [Deploy API](https://github.com/loft-br/robson_avm/actions/workflows/manual_deploy.yml) na `main` em versão major, para lançar a nova versão da API no SageMaker.
1. Teste com os resultados do novo endpoint na URL _________.

## Deploy da API - robson-deploy (API intermediária)

1. Rodar workflow [Update Stable Version](https://github.com/loft-br/robson-deploy/actions/workflows/update_stable_version.yml) com a versão do endpoint stable da nova API de produção.
1. Rodar workflow [Update Rolling Release Version](https://github.com/loft-br/robson-deploy/actions/workflows/update_rolling_version.yml)
1. Rodar workflow [Deploy to production](https://github.com/loft-br/robson-deploy/actions/workflows/deploy_to_prod.yml) da `robson-deploy`
1. Atualizar branch `develop` com todas as novidades da branch `main`. Caso haja conflitos em arquivos do DVC e `currents.yaml`, devem prevalecer versões da `main`.<br/>
   ```develop&gt; git merge main

## Batch Predict

1. Teste de batch predict puro (`--no-train`), usando a branch `release`, com todos os estimadores, usando comando da POC 7 em um laptop ou SageMaker. Rodar 2 vezes.
1. Teste de batch predict puro (`--no-train`), usando a branch `release`, com todos os estimadores, via o workflow “[Predict (✅ staging)](https://github.com/loft-br/robson_avm/actions/workflows/batch_predict_staging.yml)”.
