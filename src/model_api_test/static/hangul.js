// 한글 자모 결합 오토마타: 서버가 확정한 자모를 하나씩 받아 음절로 조합한다
const CHO  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'];
const JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'];
const JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'];
// 모델이 ㅘㅙㅝㅞ를 직접 못 내므로 기본 모음 두 개로 조합해 만든다
const V_COMBINE = {'ㅗㅏ':'ㅘ','ㅗㅐ':'ㅙ','ㅗㅣ':'ㅚ','ㅜㅓ':'ㅝ','ㅜㅔ':'ㅞ','ㅜㅣ':'ㅟ','ㅡㅣ':'ㅢ'};
const T_COMBINE = {'ㄱㅅ':'ㄳ','ㄴㅈ':'ㄵ','ㄴㅎ':'ㄶ','ㄹㄱ':'ㄺ','ㄹㅁ':'ㄻ','ㄹㅂ':'ㄼ','ㄹㅅ':'ㄽ','ㄹㅌ':'ㄾ','ㄹㅍ':'ㄿ','ㄹㅎ':'ㅀ','ㅂㅅ':'ㅄ'};
const T_SPLIT = {'ㄳ':['ㄱ','ㅅ'],'ㄵ':['ㄴ','ㅈ'],'ㄶ':['ㄴ','ㅎ'],'ㄺ':['ㄹ','ㄱ'],'ㄻ':['ㄹ','ㅁ'],'ㄼ':['ㄹ','ㅂ'],'ㄽ':['ㄹ','ㅅ'],'ㄾ':['ㄹ','ㅌ'],'ㄿ':['ㄹ','ㅍ'],'ㅀ':['ㄹ','ㅎ'],'ㅄ':['ㅂ','ㅅ']};
// 뜻 예측(최소): 모음 자리에 모음이 또 오면 앞 모음을 닮은 자음으로 재해석(ㅜㅜ→구).
// shortcut: 관측된 혼동쌍만 1:1 매핑. 오인식이 늘면 사전/빈도 기반 단어 예측으로 올린다.
const V_TO_C = {'ㅜ':'ㄱ'};

export class Hangul {
    constructor(){ this.out=''; this.c=-1; this.v=-1; this.t=0; }  // 초성/중성/종성 인덱스(t=0: 받침 없음)
    reset(){ this.out=''; this.c=-1; this.v=-1; this.t=0; }
    _render(){
        if (this.c>=0 && this.v>=0) return String.fromCharCode(0xAC00 + (this.c*21 + this.v)*28 + this.t);
        if (this.c>=0) return CHO[this.c];
        if (this.v>=0) return JUNG[this.v];
        return '';
    }
    _flush(){ this.out += this._render(); this.c=-1; this.v=-1; this.t=0; }
    text(){ return this.out + this._render(); }
    // 화면 표시용: [확정된 글자, 아직 조합 중인 음절]
    parts(){ return [this.out, this._render()]; }
    space(){ this._flush(); if (this.out && !this.out.endsWith(' ')) this.out += ' '; }
    input(j){
        const ci = CHO.indexOf(j), vi = JUNG.indexOf(j), ti = JONG.indexOf(j);
        const isV = vi >= 0;
        if (this.c<0 && this.v<0){ if (isV) this.v=vi; else this.c=ci; return; }
        if (isV){
            if (this.v<0){ this.v=vi; }                          // 초성만 있음 → 중성 결합
            else if (this.t===0){
                const comb = V_COMBINE[JUNG[this.v]+j];
                if (comb){ this.v = JUNG.indexOf(comb); }          // 복합모음
                else if (this.c<0 && V_TO_C[JUNG[this.v]]){        // 혼동 보정
                    this.c = CHO.indexOf(V_TO_C[JUNG[this.v]]); this.v = vi;
                } else { this._flush(); this.v = vi; }            // 새 음절(단독 모음)
            } else {                                              // 받침을 다음 음절 초성으로 넘김
                const split = T_SPLIT[JONG[this.t]];
                let moved;
                if (split){ moved = split[1]; this.t = JONG.indexOf(split[0]); }
                else { moved = JONG[this.t]; this.t = 0; }
                this._flush(); this.c = CHO.indexOf(moved); this.v = vi;
            }
            return;
        }
        if (this.v<0){ this._flush(); this.c = ci; }              // 자음 뒤 자음 → 분리
        else if (this.t===0){ if (ti>=0) this.t = ti; else { this._flush(); this.c = ci; } }
        else {
            const comb = T_COMBINE[JONG[this.t]+j];
            if (comb) this.t = JONG.indexOf(comb);                // 겹받침
            else { this._flush(); this.c = ci; }
        }
    }
}

// 자체 점검: 브라우저에서 주소에 ?selftest, 또는 `node src/model_api_test/static/hangul.js`
export function selftest(){
    const run = (seq) => { const h=new Hangul(); for(const j of seq) h.input(j); return h.text(); };
    const cases = [[['ㅎ','ㅏ','ㄴ','ㄱ','ㅡ','ㄹ'],'한글'],[['ㅜ','ㅜ'],'구'],
        [['ㄱ','ㅏ','ㄴ','ㅏ'],'가나'],[['ㄱ','ㅗ','ㅏ'],'과'],[['ㅇ','ㅏ','ㄴ','ㄴ','ㅕ','ㅇ'],'안녕']];
    for (const [s,e] of cases){ const g=run(s); console.assert(g===e, `자모조합 실패 ${s.join('')} -> ${g} (기대 ${e})`); }
    console.log("자모 조합 자체 점검 완료");
}
if (typeof process !== "undefined" && import.meta.url === `file://${process.argv[1]}`) selftest();
