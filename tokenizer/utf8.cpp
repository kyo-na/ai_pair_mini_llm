#include <string>
#include <vector>

std::vector<uint32_t> utf8_to_codepoints(const std::string& s){
    std::vector<uint32_t> out;
    for(size_t i=0;i<s.size();){
        uint32_t c=(unsigned char)s[i];
        if(c<0x80){ out.push_back(c); i++; }
        else if((c>>5)==0x6){
            out.push_back(((c&0x1F)<<6)|(s[i+1]&0x3F));
            i+=2;
        }else{
            out.push_back(((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F));
            i+=3;
        }
    }
    return out;
}

std::string codepoints_to_utf8(const std::vector<uint32_t>& v){
    std::string s;
    for(auto c:v){
        if(c<0x80) s.push_back((char)c);
        else if(c<0x800){
            s.push_back(0xC0|(c>>6));
            s.push_back(0x80|(c&0x3F));
        }else{
            s.push_back(0xE0|(c>>12));
            s.push_back(0x80|((c>>6)&0x3F));
            s.push_back(0x80|(c&0x3F));
        }
    }
    return s;
}