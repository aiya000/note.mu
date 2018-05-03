# 登場人物
## 矢澤にこ
　各メンバーがそれぞれのプログラミング言語極める集団、μ'ｓに所属する、スクールアイドル。

Pure Functional ProgrammingとHaskellを極めるべく、常々鍛錬している。

Haskellの話をするときは興奮気味。

# 始まり

　こんにちは、みんなのアイドル、ニコニーよ

今日はモナド上のモナドこと`MMonad`を理解することを目指すわよ

にこも`MMonad`周りは詳しくないから、一緒にやっていきましょ

にこも頑張って解説するわ！

　`MMonad`は`mmorph`パッケージ[^1]に定義されているわ

　ここでは便宜上、圏論におけるモナドを『モナド』、Haskellにおける型クラス`Monad`を『Monad』

同じく圏論におけるモナド上のモナドを『モナドモナド』[^a]、Haskellにおける方クラス`MMonad`を『MMonad』として話していくわね

例えばこんな感じね……『Haskellの`Monad`インスタンスはちょうど圏Haskのモナドになるわ』


# モナドモナドってなによ？

　`MMonad`って何かっていうと……モナドの圏の上に構築されるモナドよ

最初は圏論での意味で確認した方が簡単だから、そっちから見ていきましょうか

　まず何かが対象の普通の圏Cがあるわね、ここで一般的なモナドを考えましょう。　そう、ここのモナドは圏Cの自己関手よ

そして圏Cのモナドが対象、そのモナドの**自然変換のうち、ある条件を満たすもの**が射の圏を考えられるわ。　その圏を圏Monadsと呼びましょう

この『**ある条件を満たす、モナド間の自然変換**』を『モナドモフィズム（monad-morphism）』というの

そこで普通のモナドと同じように……圏Monads上のモナドも考えられるわね

それがつまりモナドモナドよ！

……

　だいたいわかったところで、始めていきましょうか！

# 解説の流れ

　じゃあ本題の`MMonad`について、始めましょう

以下の流れで解説していくわね

1. モナドモフィズムについて
2. `MFunctor`について
3. `MonadTrans`について
4. `MMonad`について

　Haskellのモナドモフィズムは、ある法則を満たす関数。

`MFunctor`, `MonadTrans`, `MMonad`は型クラスとして表現されるわ


# モナドモフィズムってなによ

　さて、Haskell上のモナドモフィズムとは何かを定義していくわね

Haskellの単相型が対象、関数が射の圏を圏Haskとするわ

圏Haskの定義よ。　割とよく見る一般的な圏ね。

- - -

- 圏Hask
    - 対象（単相型）
        - `Int`
        - `Char`
        - `Maybe Int`
        - `Maybe [Int]`
        - `Identity Int`
        - `State Int Char`
    - 射（関数）
        - `f :: Int -> Char`
        - `kf :: Int -> Maybe Char`
        - `Identity :: Char -> Identity Char`
            - この関数`Identity`は値構築子よ。型構築子も単なる関数にすぎないの

- - -

　圏Haskでは、ちょうど`Monad`インスタンス（`Monad`を実装したデータ型）になる`Maybe`や`Identity`、`State Int`がモナドになるわね

ここで新しい圏HighHaskを考えてみるわ

圏HighHaskでは、`Monad`インスタンスの型（種`* -> *`を持つモナド型）が対象、型`Monad m => m a -> n a`を持ち、ある法則を満たす関数が射よ

- - -

- 圏HighHask（圏Haskの`Monad`インスタンスが対象の圏）
    - 対象（`Monad`インスタンスの型）
        - `Maybe`
        - `Identity`
        - `State Int`
        - `State [Int]`
    - 射（`Monad`を別の`Monad`に変換する関数）
        - `identityToJust :: Identity a -> Maybe a`
        - `id :: Maybe a -> Maybe a`

- - -

　圏HighHaskは、Haskellで表現できる圏Monadsって感じね

どう？　わかるかしら。　この射がモナドとモナドの変換、モナドモフィズム[^b]よ

`State Int`と`State [Int]`は同じ`State a`モナドじゃないのか…って？　うんん、ここではこの2つは違うモナドとして扱われるの

`State Int a -> State [Int] a`も`State Int a -> State Char a`もモナドモフィズムね♪（ニコッ）

じゃあ、モナドモフィズムの満たすべき法則について見てみましょう！

## 文脈を届ける関数

```haskell
morph :: Monad m => m a -> n a
```

あるモナドモフィズム`morph`は、2つの法則を満たす

- 法則A
```haskell
morph (return x) = return x
```

- 法則B
```haskell
morph $ do
  x <- m
  f x
=
do
  x <- morph m
  morph $ f x
```

　何を言っているのか全然わからないって？　そうねえ、大丈夫よ。　ゆっくり考えていきましょう

まずは簡単な法則Aから考えましょ


### returnを保存する

　法則Aを難しくしているのは、異なる型の`return`が2つあるからよ

明示的に型付けしましょうか

```haskell
x        :: a
return'  :: Monad m => a -> m a
return'' :: Monad n => a -> n a
```

こうすると、こう導かれるわね

```haskell
morph :: (Monad m, Monad n) => m a -> n a
morph . return :: Monad n => a -> n a

morph . return' = return''
-- = morph (return' x) = return'' x
```

　`return`は、`Monad`の持つ副作用を起こさず、そのまま`a`を`Monad`の文脈に入れる役割を持つ関数だったわね

そうするとつまり…`morph`が`return`と合成されても、副作用を起こさないってことになるわ

mの文脈で`return`してらmをnに変換しても、nの文脈で`return`しても、同じ結果になるってことね♪

　つまるところ、以下の図式を可換にするってわけ

`return''`が副作用を起こさない関数だから、`morph`が副作用を起こさないのも自然になるわね

```
        a---------
return' |        | return''
        v        v
      m(a)---->n(a)

          morph
```

### 副作用を保存する

　今度は難しい方の法則をやるわよ

こっちも変形してから考えましょ

```haskell
morph (m >>= f) = morph m >>= morph . f
```

こっちも型を可視化するわ

```haskell
bindM :: Monad m => m a -> (a -> m b) -> m b
bindM = (>>=)
bindN :: Monad n => n c -> (c -> n d) -> n d
bindN = (>>=)

morph (m `bindM` f) = morph m `bindN` morph . f
```

　fが副作用を起こす関数だとするわ

そうした場合に、mの文脈で副作用を起こしてもnの文脈で副作用を起こしても、結果が同じになる

…って感じよ！　ねっ、言い方が難しいだけで、言ってることは簡単なのよ

## 法則

　ところで、気づいたかしら？　Haskellの型クラスMonoid、FunctorやMonadなどでも法則を満たすことを要求されたわよね

型クラスの関数の実装について要求されたわよね、例えば`Functor`だと……`fmap id = id`みたいに、`fmap`について要求されるわ

でも今回、モナドモフィズム一般について要求されているの。　つまり、あるモナドモフィズムを実装する毎に確認する必要がある

ここは注意が必要ね

　あ、でもね。　あとでまた教えるけど、モナドモフィズムを受け取る関数、`hoist`や`embed`がランク2多相なモナドモフィズムを要求しているおかげで、それについてあまり考える必要はないのよ

`hoist`や`embed`にモナドモフィズムを渡そうと思うと、自ずと自然なモナドモフィズムを作ることになるわ

- - -

　ふぅ、これでやっとモナドモフィズムが理解できたわね。
一番大きな山を超えたわ、あなたもよくがんばったわね！

ここらで少しお茶でも飲んで、休憩しましょ

# MFunctorはモナドファンクタ

　さて、ここからは実際に提供されている型クラスについてやっていくわ

まずは`MFunctor`ね。　さっきやったモナドモフィズムをそのまま扱う型クラスよ

といってもモナドモフィズムよりはまだ簡単だと思う。　あまり気を張らないでいいと思うニコ

## 射関数fmap, hoist

　`MFunctor`は以下のような型クラスよ

```haskell
class MFunctor t where
  hoist :: Monad m => (forall a. m a -> n a) -> t m b -> t n b
```

　`MFunctor`は、前述した圏HighHaskのファンクタになるの

わかりやすく言うと、『モナドファンクタ』ね

`Functor`と比較してみましょうか

```haskell
class Functor f where
  fmap :: (a -> b) -> f a -> f b
```

　`Functor`の型変数`f`が、`MFunctor`の型変数`t`に対応していそうね

じゃあ`fmap`と`hoist`がどのように対応するか

それは、これらのファンクタの射関数としての側面を見るとわかるわ

```haskell
func  :: a -> b  -- 圏Haskの射func : a -> b
morph :: Monad m => m a -> n a  -- 圏HighHaskの射（圏Haskのモナドモフィズム）morph : m -> n
```

```haskell
fmap func   :: f a -> f b      -- (a -> b)を(f a -> f b)に写した
hoist morph :: t m b -> t n b  -- (m -> n)を(t m -> t n)に写した
```

圏論のレベルで見れば、2つとも同じものってことがわかるわね

## Haskellでの表現

　同じものっていっても、Haskellが圏Hask相当のレベルで見ている以上、表現がズレちゃうの

具体的にはHaskellでは`Monad m => m -> n`っていう記法はできないわ

そのズレをうまく表現するために、`hoist`はランク2多相（Rank2Types）を使っているの

```haskell
--                   vvvvvvvvv ランク2多相
hoist :: Monad m => (forall a. m a -> n a) -> t m b -> t n b
```

　ランク2多相がわからない？　大丈夫、まだ私も完全にはわかってないから…

ここで`forall a.`が何をしているかわかればいいの

この`forall a.`はつまるところ、`m -> n`を間接的に表現しているの

`forall a. m a -> n a`の`a`を消して考えると、`m -> n`に見えると思う

　`m -> n`はモナドモフィズムよ。

モナドモフィズムは前述の通り『mとnのreturnを可換にする（returnを保存する）』 という意味で、自然な実装にする必要があったわ

　例えば、このモナドモフィズムは自然だけど

```haskell
naturalMMorph :: State Int a -> Reader Int a
naturalMMorph st = do
  let nakedSt = runState st
  reader $ fst . nakedSt
```

このモナドモフィズムは不自然よ

```haskell
unnaturalMMorph :: State Int Int -> Reader Int Int
unnaturalMMorph st = do
  let nakedSt = runState st
  reader $ (+1) . fst . nakedSt
```

理由は明白で、`unnaturalMMorph . return = return`を満たさないから

こういう不自然さは`m a -> n a`の`a`に言及しているからできることで、`hoist`の定義を

```haskell
hoist :: Monad m => (forall a. m a -> n a) -> t m b -> t n b
```

というようにすることで、引数のモナドモフィズムが`a`に言及できないようにしているわ

`hoist unnaturalMMorph`をコンパイルしようとすると、`a`が`Int`という絞られた型になっていることにより、コンパイルエラーが出るのよ

　この`forall a.`については、こちら[^2]がとてもわかりやすかったわ。　にこはここを読んで理解したニコ！

このようにして、`MFunctor`をモナドファンクタに仕立て上げているわ

…♪

# MonadTransは特性が弱まったMMonadにこよ

　ちょっと疲れてきたかしら？　でもモナドモナドまでもうもう一息！

ここまで読んでくれたあなたなら、きっとこれ以降もわかるはずよ。

がんばれ♡　がんばれ♡

　…さて今回話すこの`MonadTrans`だけど、実用上使うことも多いよく知られた、あの`MonadTrans`と全く同じものよ

実は`MonadTrans`は、`MMonad`の特性を弱めたものなの

具体的には、`MMonad`のうち`embed`と、`MFunctor`の`hoist`についての法則が要求されない

なのでモナドモフィズムも登場しないわ

　なんで特性を弱める必要があったのかって？

それはある資料[^3]によると`ContT`が`MMonad`にはなれないかららしいわよ

つまり注目すべきは、`MonadTrans`にある`lift :: (MonadTrans t, Monad m) => m a -> t m a`ね

## liftはモナドモナドのreturn

　この`lift`は`Monad`を`MonadTrans`に引き上げる関数よ。　`MMonad`については後で扱うけど、`MMonad`も`MonadTrans`なので、モナドモナドの`return`に相当するわ

意味的には`lift :: m -> t m`ね。　Haskellの`return :: a -> m a`に似てるでしょ。　高いレベルから見れば同じものよ

`lift`は以下の法則を要請するわ

```haskell
lift :: (MonadTrans t, Monad m) => m a -> t m a

lift . return = return
lift (m >>= f) = lift m >>= lift . f
```

　これ、見たことないかしら？　ちょっと考えてみて…

そう、モナドモフィズムに要請される法則と同じものなの！

```haskell
morph :: (Monad m, Monad n) => m a -> n a

morph . return :: Monad n => a -> n a
morph (m >>= f) = morph m >>= morph . f
```

　実は`lift`も、ちょうどモナドモフィズムになっているの

モナドモフィズムは`Monad m => m a -> n a`のような、`Monad`インスタンスから`Monad`インスタンスへの変換だったわね

さっき言った通り、`(MonadTrans t, Monad m) => t m`な型はモナドになるの。　だから、`lift`もモナドモフィズムよ

　`MonadTrans`インスタンスである`StateT`を例に、型を当てはめてみましょう

```haskell
instance Monad m => MonadTrans (StateT s m)

-- 簡単にするためにsとmを単相化
lift :: IO a -> StateT Int IO a
(>>=) :: a -> StateT Int IO b
```

`(>>=)`は`Monad`の代表的な演算子だし、ちょうど`(>>=) :: Monad m => a -> m b`の`m`が`StateT Int IO`になっていることがわかりやすいと思うんだけど、どうかしら

## liftはこうやって使うわ

　最後に、ついでに`lift`のオーソドックな使用例を見てみましょう

```haskell
import Control.Monad.IO.Class (MonadIO)
import Control.Monad.Trans.State.Lazy (StateT, runStateT)
import Control.Monad.Trans.Class (lift)

f :: StateT Int IO Char
f = do
  -- lift :: IO () -> StateT Int IO ()
  lift $ putStrLn "in the context of StateT Int IO"
  return 'a'

main :: IO ()
main = do
  x <- runStateT f 10
  print x

{- output -}
-- in the context of StateT Int IO
-- ('a',10)
```

`MonadTrans`は、最も自然なモナドモフィズム`lift`を持つ型クラスってところね

そう考えてもらうと、`MonadTrans`が`MMonad`を弱めたものだっていうのが、なんとなくわかるかもしれないわ

# モナモナするMMonad

　ついにたどり着いた、私たちの目的地。　モナモナするモナド、`MMonad`よ

長いようで短かったかもしれない、あなたとの勉強会もこれで終わりね

やっていきましょ

## モナドのモナド

　さて、`MMonad`の定義は以下になるわ

```haskell
class (MFunctor t, MonadTrans t) => MMonad t where
  embed :: Monad n => (forall a. m a -> t n a) -> t m b -> t n b
```

`MMonad`は`MFunctor`であって`MonadTrans`よ。　モナドが自己関手であって`MonadTrans`が`MMonad`の部分であることを鑑みれば、自然な感じね

`embed`は圏HighHaskのモナドバインドよ。　`(>>=)`と並べて見てみましょうか

```haskell
embed      :: Monad n => (forall a. m a -> t n a) -> t m b -> t n b
flip embed :: Monad n => t m b -> (forall a. m a -> t n a) -> t n b
(>>=)      :: Monad m => m a -> (a -> m b) -> m b
```

高い視点の疑似表記で見てみるわ

```haskell
flip embed :: t m -> (m -> t n) -> t n
```

```haskell
(>>=) :: m a -> (a -> m b) -> m b
```

うん。　ちゃんと一致してるわね

## モナモナする法則

　そしてやっぱり、`embed`は法則を満たす必要があるわ。　これらは`Monad`則と全く同じものよ

- 法則A
```haskell
embed lift = id
```

- 法則B
```haskell
embed f (lift m) = f m
```

- 法則C
```haskell
embed g (embed f t) = embed (\m -> embed g (f m)) t
```

`flip embed`のエイリアスとして`(|>=)`が用意されているわ。　これを使って変換してみましょう

```haskell
(|>=) :: (MMonad t, Monad n) => t m b -> (forall a. m a -> t n a) -> t n b
```

```haskell
-- A
t |>= lift  =  t
-- B
lift m |>= f  =  f m
-- C
(t |>= f) |>= g  =  t |>= (\m -> f m |>= g)
```

こう書けば、そのまま`Monad`則に対応するわ

```haskell
-- A'
m >>= return  =  m
-- B'
return a >>= k  =  k a
-- C'
(m >>= k) >>= h  =  m >>= (\x -> k x >>= h)
```

## そしてモナドモナド

　以上が`MMonad`の性質よ。

これで晴れてあなたもモナドモナドをマスターできたってこと。　よくがんばったわね

最後よに一緒に、`mmorph`に書いてある 'Embedding transformers' の章を読みましょ…

　これは実用的で、すごく面白い例よ。　にこも初めて見た時、すごく興奮したものよ

`(MonadThrow m, MonadIO m) => m a`のような型を持つ文脈でIO例外が投げられた時は、普通はそのまま`MonadThrow`値にならず…不純な例外として上位に伝播しちゃうわよね

そこで`embed`と補助関数を使えば、IO例外を`MonadThrow`値の純粋な例外として受け取れるっていうのがこの章の議題ね

　例えば`(MonadThrow m, MonadIO m) => m a`で、`f :: EitherT SomeException IO Int`に対して`x <- runEitherT f`することで`Left e`にパターンマッチできるわ

`IO`例外が`Left e`にマッピングできるのよ

```haskell
import Control.Exception.Safe (MonadThrow, try, SomeException)
import Control.Monad (join)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Morph (MMonad, embed)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Either (EitherT(..), runEitherT)

instance MMonad (EitherT e) where
  embed f m = EitherT $ do
    let nee = runEitherT . f $ runEitherT m -- :: n (Either e (Either e a))
    join <$> nee

check :: IO a -> EitherT SomeException IO a
check = EitherT . try

program :: (MonadThrow m, MonadIO m) => m String
program = liftIO $ readFile "foo"

main :: IO ()
main = do
  xOrErr <- runEitherT $ embed check program
  case xOrErr of
    Right x -> putStrLn $ "Success: " ++ x
    Left  y -> putStrLn $ "Left value is got !: " ++ show (y :: SomeException)

-- output
-- Left value is got !: foo: openFile: does not exist (No such file or directory)
```

ここに注目してちょうだい。　本質はここだけよ

```haskell
xOrErr <- runEitherT $ embed check program
```

`embed check program`の`check`で`IO`例外を`MonadThrow`例外にすくい上げてるわ

ここで見て欲しいのは、↓の例もそれと実質的にはほとんど同じであるということ

```haskell
xOrErr <- runEitherT program
```

でも`program`でfooファイルの読み込みが失敗した場合に`IO`例外が送出されて、下の`case`文が実行されなくなってしまう

その問題に対する最もシンプルな解決策としては、`program`内の`readFile`に`try`を引っ掛けることが考えられるわよね

```haskell
import Control.Exception.Safe (MonadThrow, try, SomeException, throwM, Exception)

-- てきとうに投げられる例外の定義
data AnException = AnException String
  deriving (Show)

instance Exception AnException

program :: (MonadThrow m, MonadIO m) => m String
program = do
  foo <- liftIO $ try' $ readFile "foo"
  case foo of
    Left  e -> throwM $ AnException (show e)
    Right a -> return a
  where
    -- eをSomeExceptionに推論させる
    try' :: IO String -> IO (Either SomeException String)
    try' = try

main :: IO ()
main = do
  xOrErr <- runEitherT $ embed check program
  case xOrErr of
    Right x -> putStrLn $ "Success: " ++ x
    -- IO例外もここにマッチする
    Left  y -> putStrLn $ "Left value is got !: " ++ show (y :: SomeException)
```

でもこの解法だと、すこし不安

関数`program`に余計な処理が挟まってしまっているのと、関数`program`に変更を加える必要があるってところが、ちょっとね

そういう場面で`embed`が『引っ掛けて』くれるの

こんなところに`MMonad`が使えるなんて、すごいわね

# 終わり

　ここまで読んでくれた人、ありがとう

もしあなたが「いまいち難しくてここまで飛ばし読みしてしまった」としても、それはとてもいい読み方だとわたしは思うわ

モナドモナドの面白さ、ワクワクを伝えられたなら本望よ

じゃあ、またね！

# 参考にさせてもらった資料よ

（敬称略にて）

- Monad Morphismによる局所的状態の表現（香川考司/京都大学数理解析研究所）[https://projects.repo.nii.ac.jp/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=287642&item_no=1&page_id=13&block_id=21](https://projects.repo.nii.ac.jp/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=287642&item_no=1&page_id=13&block_id=21)
- ランク2多相の、ふたつの側面（YoshikuniJujo）[http://qiita.com/YoshikuniJujo/items/c28d8fa11e33ed677e83](http://qiita.com/YoshikuniJujo/items/c28d8fa11e33ed677e83)
- モナドモナド (LT没ネタ) （hiratara）[http://qiita.com/hiratara/items/65fcf38070def7e5a918](http://qiita.com/hiratara/items/65fcf38070def7e5a918)

- - - - -

[^1]: https://www.stackage.org/haddock/lts-7.7/mmorph-1.0.6/Control-Monad-Morph.html
[^2]: http://qiita.com/YoshikuniJujo/items/c28d8fa11e33ed677e83
[^3]: http://qiita.com/hiratara/items/65fcf38070def7e5a918

[^a]: 一般的には『モナドモナド』なんて呼び方はしないらしいわ。 モナドモナドってすごい名前ね
[^b]: この射が`(Monad m, Monad n) => m a -> n a`じゃないのは、`mmorph`[^1]での定義がそうだからよ
