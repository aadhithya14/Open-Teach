/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;

namespace Oculus.Interaction
{
    public class ColliderGroup
    {
        private Collider _boundsCollider;
        private List<Collider> _colliders;

        public Collider Bounds => _boundsCollider;
        public List<Collider> Colliders => _colliders;

        public ColliderGroup(List<Collider> colliders, Collider boundsCollider)
        {
            _colliders = colliders;
            _boundsCollider = boundsCollider;
        }
    }
}
