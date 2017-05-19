# deeplearning_papers

AI論文読みリレー

* issueを使って reading listを管理しよう


1. レポジトリのクローン

```
$ git clone https://github.com/furukawa-ai/deeplearning_papers.git
```

2. ローカル編集前に必ず以下のコマンドで最新に反映すること。

```
$ git pull
```

3. 編集後は以下のコマンドでリモートをアップデート。

```
$ git add -A
$ git commit -m "<message>"
$ git push origin master
```

#### 参考: Conflictが発生した場合の解決方法（参考:http://www.backlog.jp/git-guide/pull-request/pull-request3_2.html）

```
$ git pull
```

した際に、conflictがあるファイルには、下のような箇所が出てきます

```
<<<<<<< HEAD
      if (a === b) {
=======
      if (a == b) {
>>>>>>> 839396c5383737ec06b9c2a842bfccc28f3996ef
```

これを綺麗に直します

```
       if (a === b) {
```

あとは普通にコミットしてプッシュすればOKです。

```
$ git add -A
$ git commit -m "競合を解決"
$ git push origin master
```

## 参考リンク

* [qiita 16](http://qiita.com/sakaiakira/items/9da1edda802c4884865c)
* [NIPS 16](https://nips.cc/Conferences/2016/Schedule)
* [ICLR17](http://www.iclr.cc/doku.php?id=iclr2017:conference_posters#monday_morning)
* https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap
* https://github.com/terryum/awesome-deep-learning-papers

### .gitignoreの書き方

* https://git-scm.com/docs/gitignore
